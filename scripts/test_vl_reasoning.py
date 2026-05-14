from findings.finding_generator import generate_case_findings
from reasoning.report_generator import generate_report


case_findings = generate_case_findings(
    image_path="demo_samples/cmoderatedr/b9127e38d9b9.png",
    prediction="Moderate DR",
    raw_class="cmoderatedr",
    confidence=0.4154,
    topk_predictions=[
        {"Moderate DR": 0.4154},
        {"Mild DR": 0.2512},
        {"Severe DR": 0.1821},
    ],
    cam_method="hirescam",
    cam_target_layer="stage3",
    cam_output_path="experiments/aptos_convnext_tiny/lr1e-4_bs32_seed42/explain/v0_2_2_demo",
)

report = generate_report(case_findings)
print(report)
