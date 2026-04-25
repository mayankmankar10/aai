"""
agent/agent_pipeline.py
=======================
Agent Layer — Reasoning Pipeline for Traffic Sign Classification
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from model.capsnet_model import predict_image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain knowledge — (explanation, category, safety_tip) per GTSRB class
# ---------------------------------------------------------------------------
_SIGN_KNOWLEDGE: dict = {
    0:  ("This sign indicates a maximum speed limit of 20 km/h.", "Regulatory — Speed Control", "Reduce speed; typically near schools or residential areas."),
    1:  ("This sign indicates a maximum speed limit of 30 km/h.", "Regulatory — Speed Control", "Common in urban zones; watch for pedestrians and cyclists."),
    2:  ("This sign indicates a maximum speed limit of 50 km/h.", "Regulatory — Speed Control", "Standard urban road limit; maintain safe following distance."),
    3:  ("This sign indicates a maximum speed limit of 60 km/h.", "Regulatory — Speed Control", "Typical on secondary roads; be alert for intersections."),
    4:  ("This sign indicates a maximum speed limit of 70 km/h.", "Regulatory — Speed Control", "Found on rural roads; watch for farm vehicles and animals."),
    5:  ("This sign indicates a maximum speed limit of 80 km/h.", "Regulatory — Speed Control", "Common on highways near towns; prepare to slow as you enter built-up areas."),
    6:  ("This sign marks the end of an 80 km/h speed restriction.", "Regulatory — Speed Control", "Speed restriction lifted; revert to national speed limit unless otherwise signed."),
    7:  ("This sign indicates a maximum speed limit of 100 km/h.", "Regulatory — Speed Control", "Maintain lane discipline and safe headway at high speed."),
    8:  ("This sign indicates a maximum speed limit of 120 km/h.", "Regulatory — Speed Control", "Motorway speed; stay in the right lane unless overtaking."),
    9:  ("This sign prohibits overtaking (no passing allowed).", "Regulatory — Overtaking", "Do not overtake any vehicle until the restriction ends."),
    10: ("This sign prohibits vehicles over 3.5 metric tons from overtaking.", "Regulatory — Overtaking", "Heavy goods vehicles must not overtake; passenger cars are unaffected."),
    11: ("This sign indicates right-of-way at the next intersection.", "Right-of-Way", "You have priority; other roads must yield to you."),
    12: ("This sign designates a priority road where you have right-of-way.", "Right-of-Way", "You have continuous priority; yield signs are shown on side roads."),
    13: ("This sign is used to regulate traffic by requiring drivers to yield.", "Regulatory — Yielding", "Slow down and give way to crossing traffic before proceeding."),
    14: ("This sign requires all vehicles to come to a complete stop.", "Regulatory — Stop", "You must stop fully at the stop line; proceed only when safe."),
    15: ("This sign prohibits all vehicles from entering the road.", "Prohibitory", "Entry is forbidden; find an alternative route."),
    16: ("This sign prohibits vehicles over 3.5 metric tons from entering.", "Prohibitory", "Heavy vehicles must not proceed; reroute to a suitable road."),
    17: ("This sign indicates no entry — wrong way or one-way road ahead.", "Prohibitory", "Do not enter; you are facing oncoming traffic."),
    18: ("This sign warns of a general hazard ahead.", "Warning — General", "Proceed with caution and reduce speed until the hazard is clear."),
    19: ("This sign warns of a dangerous left curve ahead.", "Warning — Road Geometry", "Reduce speed before the curve; do not brake mid-corner."),
    20: ("This sign warns of a dangerous right curve ahead.", "Warning — Road Geometry", "Reduce speed before the curve; do not brake mid-corner."),
    21: ("This sign warns of a double curve ahead.", "Warning — Road Geometry", "Slow down significantly; double curves require multiple steering adjustments."),
    22: ("This sign warns of a bumpy or uneven road surface ahead.", "Warning — Road Surface", "Reduce speed to avoid vehicle damage and maintain control."),
    23: ("This sign warns that the road surface is slippery.", "Warning — Road Surface", "Reduce speed, increase following distance, avoid sudden manoeuvres."),
    24: ("This sign warns that the road narrows on the right side.", "Warning — Road Geometry", "Move towards the centre; be prepared to give way to oncoming traffic."),
    25: ("This sign warns of road construction or maintenance work ahead.", "Warning — Road Work", "Reduce speed, follow temporary signals, and watch for workers."),
    26: ("This sign warns that traffic signals are ahead.", "Warning — Traffic Control", "Be prepared to stop; approach with caution if signals are flashing."),
    27: ("This sign warns of a pedestrian crossing ahead.", "Warning — Pedestrians", "Reduce speed and be prepared to stop for pedestrians."),
    28: ("This sign warns of a children crossing zone ahead.", "Warning — Pedestrians", "Slow down significantly; children may enter the road unexpectedly."),
    29: ("This sign warns of a bicycle crossing ahead.", "Warning — Cyclists", "Watch for cyclists crossing; give them sufficient space."),
    30: ("This sign warns of ice or snow on the road ahead.", "Warning — Road Surface", "Reduce speed drastically; use winter tyres or chains if required."),
    31: ("This sign warns that wild animals may cross the road.", "Warning — Animals", "Reduce speed especially at dawn and dusk; scan the road edges."),
    32: ("This sign marks the end of all speed and passing restrictions.", "Regulatory — End of Restrictions", "National speed and passing rules now apply."),
    33: ("This sign instructs drivers to turn right ahead.", "Mandatory — Direction", "You must turn right at the indicated point; signal in advance."),
    34: ("This sign instructs drivers to turn left ahead.", "Mandatory — Direction", "You must turn left at the indicated point; signal in advance."),
    35: ("This sign instructs drivers to proceed straight ahead only.", "Mandatory — Direction", "No turning permitted; continue forward."),
    36: ("This sign permits going straight or turning right.", "Mandatory — Direction", "Choose either direction; signal your intended path."),
    37: ("This sign permits going straight or turning left.", "Mandatory — Direction", "Choose either direction; signal your intended path."),
    38: ("This sign instructs drivers to keep right.", "Mandatory — Positioning", "Stay on the right side of the obstruction or divider."),
    39: ("This sign instructs drivers to keep left.", "Mandatory — Positioning", "Stay on the left side of the obstruction or divider."),
    40: ("This sign indicates a mandatory roundabout; follow the circular route.", "Mandatory — Roundabout", "Yield to traffic already in the roundabout; signal when exiting."),
    41: ("This sign marks the end of the no-passing zone.", "Regulatory — End of Restriction", "Overtaking is now permitted where safe and legal."),
    42: ("This sign marks the end of the no-passing restriction for heavy vehicles.", "Regulatory — End of Restriction", "Heavy vehicles may now overtake where safe and legal."),
}

_FALLBACK_KNOWLEDGE = (
    "This sign has been classified by the CapsNet model.",
    "Traffic Sign",
    "Observe and respond to this sign according to local traffic regulations.",
)

_HIGH_CONFIDENCE = 0.80
_MEDIUM_CONFIDENCE = 0.50


def _interpret_confidence(confidence: float) -> str:
    if confidence >= _HIGH_CONFIDENCE:
        return f"High confidence ({confidence:.1%}). The model is very certain about this classification."
    elif confidence >= _MEDIUM_CONFIDENCE:
        return (f"Moderate confidence ({confidence:.1%}). The prediction is likely correct, "
                "but lighting or occlusion may affect accuracy.")
    else:
        return (f"Low confidence ({confidence:.1%}). The model is uncertain — verify visually "
                "or provide a higher-resolution image.")


@dataclass
class AgentResponse:
    """Structured output from the agent pipeline."""
    success: bool = False
    class_index: int = -1
    label: str = "Unknown"
    confidence: float = 0.0
    confidence_assessment: str = ""
    category: str = ""
    explanation: str = ""
    safety_tip: str = ""
    reasoning_trace: list = field(default_factory=list)
    error_message: str = ""


def agent_pipeline(image_path: str, model_path: Optional[str] = None) -> AgentResponse:
    """
    Five-step agentic reasoning pipeline for traffic sign classification.

    Steps:
      1. Input validation
      2. Perception — CapsNet inference
      3. Knowledge retrieval — domain knowledge lookup
      4. Confidence interpretation
      5. Response synthesis
    """
    response = AgentResponse()
    trace = response.reasoning_trace

    # Step 1 — Input Validation
    trace.append("Step 1 | Input Validation: Checking image path.")
    if not image_path or not isinstance(image_path, str):
        response.error_message = "No image path provided."
        trace.append("  ✗ Validation failed: image_path is empty or not a string.")
        return response
    trace.append(f"  ✓ Image path accepted: '{image_path}'")

    # Step 2 — Perception
    trace.append("Step 2 | Perception: Invoking CapsNet model for classification.")
    try:
        class_index, confidence, label = predict_image(image_path, model_path)
        response.class_index = class_index
        response.confidence = confidence
        response.label = label
        trace.append(f"  ✓ class_index={class_index}, confidence={confidence:.4f}, label='{label}'")
    except FileNotFoundError as exc:
        response.error_message = f"Image file error: {exc}"
        trace.append(f"  ✗ FileNotFoundError: {exc}")
        return response
    except ValueError as exc:
        response.error_message = f"Image decoding error: {exc}"
        trace.append(f"  ✗ ValueError: {exc}")
        return response
    except RuntimeError as exc:
        response.error_message = f"Model inference error: {exc}"
        trace.append(f"  ✗ RuntimeError: {exc}")
        return response
    except Exception as exc:
        response.error_message = f"Unexpected error: {exc}"
        trace.append(f"  ✗ Unexpected: {exc}")
        logger.exception("Unexpected error in agent_pipeline")
        return response

    # Step 3 — Knowledge Retrieval
    trace.append("Step 3 | Knowledge Retrieval: Looking up domain knowledge.")
    explanation, category, safety_tip = _SIGN_KNOWLEDGE.get(class_index, _FALLBACK_KNOWLEDGE)
    response.explanation = explanation
    response.category = category
    response.safety_tip = safety_tip
    trace.append(f"  ✓ Category: '{category}'")

    # Step 4 — Confidence Interpretation
    trace.append("Step 4 | Confidence Interpretation: Assessing model certainty.")
    response.confidence_assessment = _interpret_confidence(confidence)
    trace.append(f"  ✓ {response.confidence_assessment}")

    # Step 5 — Response Synthesis
    trace.append("Step 5 | Response Synthesis: Assembling final structured response.")
    response.success = True
    trace.append("  ✓ Agent pipeline completed successfully.")

    logger.info("AgentResponse ready — label='%s', confidence=%.4f", response.label, response.confidence)
    return response
