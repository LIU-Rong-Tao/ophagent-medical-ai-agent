import torch
import torch.nn.functional as F
from torchvision import transforms

from agent.schema import AgentInput, AgentResult, TopKPrediction
from agent.providers import generate_agent_report
from findings.finding_generator import generate_case_findings


def build_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def run_agent(
    agent_input: AgentInput,
    *,
    model,
    device,
    idx_to_class,
    image_size,
    class_display_names=None,
):
    """
    Lightweight OphAgent workflow runner.

    image -> classification -> top-k -> findings -> report -> AgentResult
    """

    class_display_names = class_display_names or {}

    transform = build_transform(image_size)

    input_tensor = (
        transform(agent_input.image.convert("RGB"))
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)[0]

    probs = probs.detach().cpu()

    top_probs, top_indices = torch.topk(
        probs,
        k=min(agent_input.top_k, len(probs)),
    )

    topk_predictions = []

    for rank, (idx, prob) in enumerate(
        zip(top_indices.tolist(), top_probs.tolist()),
        start=1,
    ):
        raw_class = idx_to_class[int(idx)]

        display_name = class_display_names.get(
            raw_class,
            raw_class,
        )

        topk_predictions.append(
            TopKPrediction(
                rank=rank,
                raw_class=raw_class,
                display_name=display_name,
                confidence=float(prob),
            )
        )

    top1 = topk_predictions[0]

    predicted_class = top1.raw_class
    predicted_display_name = top1.display_name
    confidence = float(top1.confidence)

    case_findings = generate_case_findings(
        image_path=agent_input.image_source,
        prediction=predicted_display_name,
        raw_class=predicted_class,
        confidence=confidence,
        topk_predictions=[
            {item.display_name: float(item.confidence)}
            for item in topk_predictions
        ],
        cam_method=None,
        cam_target_layer=None,
        cam_output_path=None,
    )

    report_result = generate_agent_report(
        findings=case_findings,
        providers=agent_input.report_providers,
        fallback_provider=agent_input.fallback_report_provider,
        report_config=agent_input.report_config,
    )

    return AgentResult(
        predicted_class=predicted_class,
        predicted_display_name=predicted_display_name,
        confidence=confidence,
        topk=topk_predictions,
        findings=case_findings,
        report=report_result.report,
        report_provider=report_result.provider,
        raw={
            "image_source": agent_input.image_source,
            "probs": probs.tolist(),
        },
    )