import logging
from typing import Dict, Any

logger = logging.getLogger("MLInferenceService")

class MLInferenceService:
    async def validate_violation_frame(self, frame_path: str) -> Dict[str, Any]:
        """
        Mock CV module that validates if a saved frame contains a vehicle violation.
        Used by the API to verify uploaded snapshots as a sanity check.
        """
        logger.info(f"Running ML verification on frame snapshot: {frame_path}")
        
        # Simple mock response
        return {
            "validation_status": "SUCCESS",
            "confidence": 0.96,
            "detected_objects": ["car", "license_plate"],
            "violation_verified": True
        }

ml_inference_service = MLInferenceService()
