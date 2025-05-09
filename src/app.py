import gradio as gr
import matplotlib.pyplot as plt
import p2
number_of_differences = 5
# Set the page title for the Gradio app
def preview_and_process(image, num_differences):
    # プレビュー用の画像をそのまま返す
    # 実際の処理ロジックをここに追加可能
    i2 = p2.pil2cv(image)
    orig, modified, composite_on_paper, answer = p2.generate_difference_pair(i2, num_differences)
    # 画像をPIL形式に変換して返す
    i3 = p2.cv2pil(composite_on_paper)
    answer = p2.cv2pil(answer)
    return i3, answer

with gr.Blocks() as demo:
    demo.title = "Spot the Difference Generator"
    gr.Markdown("<h1 style='text-align: center;'>間違い探し生成</h1>")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="画像をアップロード", type="filepath")
            number_of_differences_slider = gr.Slider(minimum=1, maximum=10, value=5, label="間違いの数", step=1, interactive=True)
            submit_button = gr.Button("生成！")
            image_preview = gr.Image(label="出力", interactive=False)
            answer_image = gr.Image(label="答え", interactive=False)
    
    submit_button.click(lambda image, num: preview_and_process(image, num), inputs=[image_input, number_of_differences_slider], outputs=[image_preview, answer_image])

demo.launch()
