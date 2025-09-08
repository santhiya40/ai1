import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
model_name = "ibm-granite/granite-3.2-2b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def generate_response(prompt, max_length=1024):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(prompt, "").strip()
    return response

    ity_analysis(city_name)
    rompt= f"provide a detailed analysis of {city_name} including:\n1.crime index and safety statistics \n2.accident rates and traffic safety information\n3.overall safety assessment\n\ncity:{city}"
    return generate_response(prompt, max_length=1000)

    itizen_interfaction(query)
    rompt = f"as a government assisstant,provide accurate and helpfull information about the following citizen query related to public serives,government policies,or civic issues:\n\nQuerty:{query}"
    return generate_response(prompt, max_length=1000)

# ate Gradio interface
with gr.Blocks() as app:
    gr.Markdown("# Eco Assistant & Policy Analyzer")

    with gr.Tabs():
        with gr.TabItem("Eco Tips Generator"):
            with gr.Row():
                with gr.Column():
                    keywords_input = gr.Textbox(
                        label="enter city name",
                        placeholder="e.g., New york,london, mumbai...",
                        lines=3
                    )
                    analyze_btn = gr.Button("Analyze city")

                with gr.Column():
                    city_output = gr.Textbox(label="city analysis(crime index & accidents)", lines=15)

            analyze_btn.click(city_analysis, inputs=city_input, outputs=city_output)

        with gr.TabItem("citizen serivices"):
            with gr.Row():
                with gr.Column():
                  citizen_query = gr.textbox()
                      label="your Query",
                      placeholder+"ask about public services,government policies,civic issues...",
                      lines=4
                   query_btn = gr.button("get information")
                with gr.Column():
                     citizen_output = gr.textbox(label="government response",lines=15)
            query_btn.click(citizen_interaction,inputs=citizen_query,outputs=citizen_output)
            app.launch(share=True)