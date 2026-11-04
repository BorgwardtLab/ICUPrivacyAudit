import gc
import logging
import hydra
import joblib
import torch
import lightning as L
from omegaconf import OmegaConf, DictConfig
from syntheticicu.resolvers import register_resolvers
from syntheticicu.demo_utils import (
    prepare_demographics,
    train_demographic_model,
    resolve_demo_model_cfg,
)
from hydra import initialize, compose
import sys
import time
import numpy as np
import os

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

register_resolvers()

log = logging.getLogger(__name__)


def get_model_class(cfg):
    """Get the appropriate model class based on config."""
    model_target = getattr(
        cfg.model, "_target_", "syntheticicu.model_emb.get_llama_model"
    )
    if "model_gap" in model_target:
        from syntheticicu.model_gap import ICULlamaGap

        return ICULlamaGap
    else:
        from syntheticicu.model_emb import ICULlama

        return ICULlama


def main():
    config_name = sys.argv[1] if len(sys.argv) > 1 else "generate"
    overrides = sys.argv[2:]

    with initialize(config_path="../config", version_base="1.3"):
        cfg = compose(config_name=config_name, overrides=overrides)
        print(OmegaConf.to_yaml(cfg), flush=True)

    log.info(f"Configs:\n{OmegaConf.to_yaml(cfg)}")
    L.seed_everything(cfg.seed, workers=True)

    model = load_model(cfg)
    print(len(model._datamodule.test_icu_dataset), flush=True)
    log_environment()
    evaluate_held_out(cfg, model)

    demo_generator = build_demographic_generator(cfg, model)
    output_tag = cfg.sampling.output_tag
    # per-batch results land here so a timed-out job resumes where it stopped
    checkpoint_dir = f"{cfg.logs.path}/gen_chunks_{output_tag}"
    start = time.time()
    with torch.no_grad():
        corpus = model.generate_batched(
            num_samples_per_class=cfg.sampling.num_samples_per_class,
            use_real_prevalence=cfg.sampling.use_real_prevalence,
            cfg_scale=cfg.sampling.cfg_scale,
            batch_size=cfg.sampling.get("batch_size", 32),
            include_demographics=cfg.sampling.include_demographics,
            gen_demo=demo_generator,
            checkpoint_dir=checkpoint_dir,
            gpu_count=int(cfg.sampling.gpu_count),
        )
    print("Generation time:", time.time() - start, flush=True)
    save_outputs(cfg, model._datamodule.tokenizer, corpus, checkpoint_dir)


def load_model(cfg):
    """The checkpoint at model.pretrained_path, in eval mode, on the GPU if any."""
    log.info(f"Loading model from {cfg.model.pretrained_path}...")
    ModelClass = get_model_class(cfg)
    log.info(f"Using model class: {ModelClass.__name__}")
    model = ModelClass.load_from_checkpoint(
        cfg.model.pretrained_path, new_cfg=cfg, weights_only=False
    )
    if torch.cuda.is_available():
        model.to("cuda")
    model.eval()
    return model


def log_environment():
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    print(f"CUDA device count: {torch.cuda.device_count()}", flush=True)
    print(
        f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}",
        flush=True,
    )
    print(f"SLURM_PROCID: {os.environ.get('SLURM_PROCID', 'not set')}", flush=True)
    print(f"SLURM_NTASKS: {os.environ.get('SLURM_NTASKS', 'not set')}", flush=True)
    print(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'not set')}", flush=True)


def evaluate_held_out(cfg, model):
    """Test-set (or validation) metrics of the checkpoint, logged per corpus."""
    datamodule = model._datamodule
    logger = []
    # one log dir per corpus, so corpora generated in parallel into the same
    # logs.path do not race on csv_logs/version_N
    logger.append(
        L.pytorch.loggers.CSVLogger(
            cfg.logs.path, name=f"csv_logs_{cfg.sampling.output_tag}"
        )
    )
    trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    if hasattr(datamodule, "test_icu_dataset"):
        trainer.test(model, datamodule)
    else:
        trainer.validate(model, datamodule)

    # Lightning's teardown moves the model back to CPU after test/validate
    if torch.cuda.is_available():
        model.to("cuda")
    print(f"Generating on device: {model.device}", flush=True)


def build_demographic_generator(cfg, model):
    """Tabular demographics model trained on the train split, or None.

    None makes the LLaMA model generate the demographics itself.
    """
    gen_demo_cfg = cfg.sampling.gen_demo
    if gen_demo_cfg is None:
        return None
    _, _, demo_df = prepare_demographics(
        model._datamodule.return_label, model._datamodule.train_icu_dataset
    )
    return train_demographic_model(
        gen_model_cfg=resolve_demo_model_cfg(gen_demo_cfg),
        df_train=demo_df,
        model_path=cfg.sampling.get("model_demo_path", None),
    )


def save_outputs(cfg, tokenizer, corpus, checkpoint_dir):
    """Write the corpus (and attentions / logits when requested) to logs.path.

    Every tensor goes to the CPU first: a CUDA tensor in a saved corpus makes it
    unloadable on CPU-only nodes.
    """
    (
        generated_samples,
        generated_labels,
        generated_centers,
        generated_attentions,
        generated_logits,
    ) = corpus
    output_tag = cfg.sampling.output_tag
    if torch.is_tensor(generated_labels):
        generated_labels = generated_labels.cpu()
    if torch.is_tensor(generated_centers):
        generated_centers = generated_centers.cpu()
    assert len(generated_samples) == len(generated_labels)
    assert len(generated_samples) == len(generated_centers)

    # Move logits to CPU if they exist (can be very large)
    if generated_logits:
        generated_logits = [
            score.cpu() if torch.is_tensor(score) else score
            for score in generated_logits
        ]

    # Force garbage collection and clear CUDA cache
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[Memory cleanup] Moved tensors to CPU, cleared CUDA cache", flush=True)

    joblib.dump(
        (generated_samples, generated_labels, generated_centers),
        f"{cfg.logs.path}/generated_samples_{output_tag}.joblib",
    )
    print(f"Per-batch chunks kept in {checkpoint_dir} (safe to delete)", flush=True)
    if cfg.sampling.output_attentions:
        attentions_cpu = []
        for att in generated_attentions:
            if isinstance(att, list):
                attentions_cpu.append(
                    [a.cpu() if torch.is_tensor(a) else a for a in att]
                )
            elif torch.is_tensor(att):
                attentions_cpu.append(att.cpu())
            else:
                attentions_cpu.append(att)

        joblib.dump(attentions_cpu, cfg.logs.path + "/attentions.joblib")
        del attentions_cpu, generated_attentions
        gc.collect()

    if cfg.sampling.output_scores:
        joblib.dump(
            generated_logits,
            f"{cfg.logs.path}/logits_{output_tag}.joblib",
        )
        joblib.dump(tokenizer.vocab, cfg.logs.path + "/ftoi.joblib")
        joblib.dump(tokenizer.id_to_token, cfg.logs.path + "/itof.joblib")

    del generated_logits
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
