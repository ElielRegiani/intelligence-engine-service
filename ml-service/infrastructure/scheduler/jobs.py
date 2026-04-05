"""Daily jobs: training (18:20) and batch inference (18:30)."""

from __future__ import annotations

import logging
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from app.services.prediction_service import PredictionService
from infrastructure.config.settings import Settings, get_settings
from training.pipelines.training_pipeline import TrainingPipeline

logger = logging.getLogger(__name__)

_scheduler: Optional[BackgroundScheduler] = None


def run_training_job() -> None:
    settings = get_settings()
    logger.info("scheduled_training_start")
    try:
        pipeline = TrainingPipeline(settings=settings)
        result = pipeline.run()
        logger.info(
            "scheduled_training_done",
            extra={
                "accuracy": result.accuracy,
                "model": result.model_filename,
                "seconds": round(result.duration_seconds, 3),
            },
        )
    except Exception as e:
        logger.exception("scheduled_training_failed: %s", e)


def run_batch_inference_job() -> None:
    settings = get_settings()
    logger.info("scheduled_batch_inference_start")
    svc = PredictionService(settings=settings)
    for sym in settings.batch_symbol_list:
        try:
            svc.predict(sym, publish_event=True)
        except Exception as e:
            logger.exception("batch_predict_failed %s: %s", sym, e)


def setup_scheduler(settings: Optional[Settings] = None) -> Optional[BackgroundScheduler]:
    global _scheduler
    s = settings or get_settings()
    if not s.scheduler_enabled:
        return None
    if _scheduler is not None:
        return _scheduler

    sched = BackgroundScheduler(timezone=s.scheduler_timezone)
    sched.add_job(
        run_training_job,
        CronTrigger(hour=s.train_cron_hour, minute=s.train_cron_minute),
        id="daily_training",
        replace_existing=True,
    )
    sched.add_job(
        run_batch_inference_job,
        CronTrigger(hour=s.batch_cron_hour, minute=s.batch_cron_minute),
        id="daily_batch_inference",
        replace_existing=True,
    )
    sched.start()
    _scheduler = sched
    logger.info(
        "scheduler_started",
        extra={
            "train": f"{s.train_cron_hour:02d}:{s.train_cron_minute:02d}",
            "batch": f"{s.batch_cron_hour:02d}:{s.batch_cron_minute:02d}",
        },
    )
    return _scheduler


def shutdown_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
