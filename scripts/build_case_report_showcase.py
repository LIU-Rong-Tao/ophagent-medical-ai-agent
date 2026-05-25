from pathlib import Path
import json
import textwrap

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    if bold:
        candidates = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
        ] + candidates

    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)

    return ImageFont.load_default()


def resize_keep_aspect(img: Image.Image, target_w: int, target_h: int):
    img = img.convert("RGB")
    img.thumbnail((target_w, target_h))
    canvas = Image.new("RGB", (target_w, target_h), "white")
    x = (target_w - img.width) // 2
    y = (target_h - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def draw_wrapped(draw, text, xy, font, fill, width_chars, line_gap=8):
    x, y = xy
    lines = []

    for paragraph in text.split("\n"):
        if not paragraph.strip():
            lines.append("")
            continue

        lines.extend(textwrap.wrap(paragraph, width=width_chars))

    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += font.size + line_gap

    return y


def draw_status_line(draw, x, y, label, value, font, ok=True):
    mark = "✓" if ok else "!"
    color = "#047857" if ok else "#b91c1c"
    draw.text((x, y), mark, font=font, fill=color)
    draw.text((x + 34, y), f"{label}：{value}", font=font, fill="#111827")
    return y + font.size + 12


def main():
    case_dir = PROJECT_ROOT / "experiments/case_reports/d9bbdc33db83"
    output_path = PROJECT_ROOT / "docs/assets/v0_6_case_report_showcase.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_img = Image.open(case_dir / "input.png")
    overlay_img = Image.open(case_dir / "cam/overlay.png")

    with open(case_dir / "prediction.json", "r", encoding="utf-8") as f:
        prediction = json.load(f)

    with open(case_dir / "validation.json", "r", encoding="utf-8") as f:
        validation = json.load(f)

    W, H = 1900, 980
    canvas = Image.new("RGB", (W, H), "#f8fafc")
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(44, bold=True)
    subtitle_font = load_font(25)
    section_font = load_font(31, bold=True)
    text_font = load_font(24)
    small_font = load_font(20)

    draw.text((50, 35), "OphAgent v0.6.0 证据瓶颈病例报告原型", font=title_font, fill="#111827")
    draw.text(
        (50, 95),
        "眼底图像 → 模型预测 → CAM 弱视觉证据 → 结构化发现 → 声明级验证 → 报告草稿",
        font=subtitle_font,
        fill="#374151",
    )

    card_y = 160
    card_h = 720
    gap = 30
    card_w = (W - 100 - 2 * gap) // 3

    cards = [
        (50, card_y, card_w, card_h, "输入眼底图像"),
        (50 + card_w + gap, card_y, card_w, card_h, "CAM 弱视觉证据"),
        (50 + 2 * (card_w + gap), card_y, card_w, card_h, "报告与验证摘要"),
    ]

    for x, y, w, h, title in cards:
        draw.rounded_rectangle(
            (x, y, x + w, y + h),
            radius=24,
            fill="white",
            outline="#e5e7eb",
            width=2,
        )
        draw.text((x + 25, y + 22), title, font=section_font, fill="#111827")

    img_w = card_w - 80
    img_h = 500

    input_resized = resize_keep_aspect(input_img, img_w, img_h)
    overlay_resized = resize_keep_aspect(overlay_img, img_w, img_h)

    x1, y1, _, _, _ = cards[0]
    x2, y2, _, _, _ = cards[1]
    x3, y3, _, _, _ = cards[2]

    canvas.paste(input_resized, (x1 + 40, y1 + 95))
    canvas.paste(overlay_resized, (x2 + 40, y2 + 95))

    draw.text((x1 + 40, y1 + 620), "原始眼底图像", font=text_font, fill="#374151")
    draw.text((x1 + 40, y1 + 655), "样本目录标签：cmoderatedr", font=text_font, fill="#374151")
    draw.text((x1 + 40, y1 + 690), "说明：目录标签仅用于样本追踪，不代表临床诊断", font=small_font, fill="#6b7280")

    draw.text((x2 + 40, y2 + 620), "CAM 仅表示模型关注区域", font=text_font, fill="#374151")
    draw.text((x2 + 40, y2 + 655), "不是病灶标注，也不是临床定位依据", font=text_font, fill="#b91c1c")

    right_x = x3 + 35
    cur_y = y3 + 95

    draw.text((right_x, cur_y), "模型输出", font=section_font, fill="#111827")
    cur_y += 52

    draw.text(
        (right_x, cur_y),
        f"预测倾向：{prediction['display_name']}",
        font=text_font,
        fill="#111827",
    )
    cur_y += 36

    draw.text(
        (right_x, cur_y),
        f"置信度：{prediction['confidence']:.4f}",
        font=text_font,
        fill="#111827",
    )
    cur_y += 36

    if len(prediction["topk_predictions"]) > 1:
        top2 = prediction["topk_predictions"][1]
        draw.text(
            (right_x, cur_y),
            f"第二高置信度类别：{top2['display_name']} ({top2['confidence']:.4f})",
            font=text_font,
            fill="#111827",
        )
        cur_y += 48

    draw.text((right_x, cur_y), "报告验证", font=section_font, fill="#111827")
    cur_y += 50

    cur_y = draw_status_line(
        draw,
        right_x,
        cur_y,
        "结构校验通过",
        str(validation["schema_valid"]),
        text_font,
        ok=validation["schema_valid"],
    )

    cur_y = draw_status_line(
        draw,
        right_x,
        cur_y,
        "无证据支撑声明数",
        str(validation["unsupported_claim_count"]),
        text_font,
        ok=validation["unsupported_claim_count"] == 0,
    )

    cur_y = draw_status_line(
        draw,
        right_x,
        cur_y,
        "证据覆盖率",
        str(validation["evidence_coverage_rate"]),
        text_font,
        ok=validation["evidence_coverage_rate"] == 1.0,
    )

    cur_y = draw_status_line(
        draw,
        right_x,
        cur_y,
        "越权诊断声明",
        str(validation["clinical_diagnosis_claim_present"]),
        text_font,
        ok=not validation["clinical_diagnosis_claim_present"],
    )

    cur_y = draw_status_line(
        draw,
        right_x,
        cur_y,
        "图像质量过度声明",
        str(validation["image_quality_overclaimed"]),
        text_font,
        ok=not validation["image_quality_overclaimed"],
    )

    cur_y += 25
    draw.text((right_x, cur_y), "安全边界", font=section_font, fill="#111827")
    cur_y += 50

    safety_text = (
        "仅科研/展示用途\n"
        "不能用于临床诊断或治疗建议\n"
        "CAM 是弱视觉证据，不是病灶标注\n"
        "所有输出需要人工审核"
    )

    draw_wrapped(
        draw,
        safety_text,
        (right_x, cur_y),
        text_font,
        "#374151",
        width_chars=30,
        line_gap=10,
    )

    draw.text(
        (50, 920),
        "输出报告：experiments/case_reports/d9bbdc33db83/report.html",
        font=small_font,
        fill="#4b5563",
    )

    canvas.save(output_path)
    print(f"Saved showcase image: {output_path}")


if __name__ == "__main__":
    main()