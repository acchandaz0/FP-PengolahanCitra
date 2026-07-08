"""Demo Gradio — Segmentasi Glioma Otak MMSK-3D U-Net.

Layout: pilih bidang (Aksial/Koronal/Sagital), geser slider menelusuri irisan,
lihat tiga panel berdampingan: Raw | Ground Truth | Hasil Segmentasi.

Jalankan:  python app.py
"""
import gradio as gr
import numpy as np

import config
import inference
import visualization as vis

MODALITY_IDX = {m: i for i, m in enumerate(config.MODALITY_ORDER)}
COLOR_LABEL = "Per-label (NCR/ED/NET/ET)"
COLOR_REGION = "Per-region (WT/TC/ET)"


def _overlay(img_slice, seg_slice, coloring):
    if coloring == COLOR_REGION:
        return vis.overlay_per_region(img_slice, seg_slice)
    return vis.overlay_per_label(img_slice, seg_slice)


def update(case_name, z, modality, coloring, plane, model_name):
    case = inference.get_case(case_name, model_name)
    images, seg, pred = case["images"], case["seg"], case["pred"]
    z = int(z)
    ch = MODALITY_IDX[modality]

    img_slice = vis.take_slice(images[ch], plane, z)
    seg_slice = vis.take_slice(seg, plane, z)

    raw_img = vis.render_raw(img_slice)
    gt_img = _overlay(img_slice, seg_slice, coloring)

    if pred is not None:
        pred_slice = vis.take_slice(pred, plane, z)
        pr_img = _overlay(img_slice, pred_slice, coloring)
        d = case["dice"]
        dice_md = (
            f"**Dice (kasus ini)** &nbsp;&nbsp; "
            f"WT `{d['WT']:.3f}` &nbsp; TC `{d['TC']:.3f}` &nbsp; ET `{d['ET']:.3f}`"
        )
    else:
        pr_img = np.zeros_like(raw_img)
        dice_md = (
            f"_Prediksi untuk **{model_name}** belum tersedia. Isi path checkpoint "
            f"model ini pada `MODELS` di `config.py` untuk menampilkannya._"
        )

    return raw_img, gt_img, pr_img, dice_md


def reset_and_update(case_name, modality, coloring, plane, model_name):
    """Saat ganti plane atau kasus: lompat ke irisan paling informatif bidang itu."""
    seg = inference.get_case(case_name, model_name)["seg"]
    z = inference.best_slice(seg, plane)
    raw_img, gt_img, pr_img, dice_md = update(case_name, z, modality, coloring, plane, model_name)
    return gr.update(value=z), raw_img, gt_img, pr_img, dice_md


def _legend_html(coloring):
    items = config.REGION_COLORS.items() if coloring == COLOR_REGION else \
        [(config.LABEL_NAMES[l], c) for l, c in config.LABEL_COLORS.items()]
    spans = " &nbsp;&nbsp; ".join(
        f'<span style="color:rgb{c};font-size:18px">&#9632;</span> {name}'
        for name, c in items
    )
    return f'<div style="font-size:14px">{spans}</div>'


with gr.Blocks(title="Segmentasi Glioma — MMSK-3D U-Net") as demo:
    gr.Markdown("## Segmentasi Glioma Otak — MMSK-3D U-Net")
    gr.Markdown(
        "Pilih bidang pandang dan skema warna, lalu geser slider untuk menelusuri "
        "irisan. Tiga panel: citra mentah, ground truth, dan hasil segmentasi model."
    )

    model = gr.Radio(
        list(config.MODELS), value=config.DEFAULT_MODEL, label="Model"
    )

    with gr.Row():
        case = gr.Dropdown(
            list(config.SAMPLE_CASES), value=list(config.SAMPLE_CASES)[0], label="Kasus"
        )
        plane = gr.Radio(vis.PLANES, value="Aksial", label="Bidang (viewpoint)")
        modality = gr.Radio(
            config.MODALITY_ORDER, value=config.DEFAULT_MODALITY, label="Modalitas latar"
        )
        coloring = gr.Radio(
            [COLOR_LABEL, COLOR_REGION], value=COLOR_LABEL, label="Skema warna"
        )

    slider = gr.Slider(0, 127, step=1, value=0, label="Irisan")
    legend = gr.HTML(_legend_html(COLOR_LABEL))

    with gr.Row(equal_height=True):
        raw_out = gr.Image(label="Raw", type="numpy", height=440)
        gt_out = gr.Image(label="Ground Truth", type="numpy", height=440)
        pr_out = gr.Image(label="Hasil Segmentasi", type="numpy", height=440)

    dice_out = gr.Markdown()

    render_inputs = [case, slider, modality, coloring, plane, model]
    render_outputs = [raw_out, gt_out, pr_out, dice_out]
    reset_inputs = [case, modality, coloring, plane, model]
    reset_outputs = [slider, raw_out, gt_out, pr_out, dice_out]

    # geser slider / ganti modalitas / warna / model -> render ulang di slice yang sama
    slider.change(update, render_inputs, render_outputs)
    modality.change(update, render_inputs, render_outputs)
    coloring.change(update, render_inputs, render_outputs)
    coloring.change(_legend_html, coloring, legend)
    model.change(update, render_inputs, render_outputs)

    # ganti plane / kasus -> reset slider ke irisan paling informatif lalu render
    plane.change(reset_and_update, reset_inputs, reset_outputs)
    case.change(reset_and_update, reset_inputs, reset_outputs)

    def _init():
        inference.warmup()
        first = list(config.SAMPLE_CASES)[0]
        seg = inference.get_case(first, config.DEFAULT_MODEL)["seg"]
        z = inference.best_slice(seg, "Aksial")
        raw_img, gt_img, pr_img, dice_md = update(
            first, z, config.DEFAULT_MODALITY, COLOR_LABEL, "Aksial", config.DEFAULT_MODEL
        )
        return gr.update(value=z), raw_img, gt_img, pr_img, dice_md

    demo.load(_init, None, [slider, raw_out, gt_out, pr_out, dice_out])


if __name__ == "__main__":
    # share=True untuk link sementara yang bisa dibuka dari laptop lain saat sidang
    demo.launch()
