import json
import os
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from termcolor import colored
import torch
import numpy as np
from tkinter import messagebox
import tkinter as tk

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

class CodeGenerator:
    def __init__(self, model_name="codellama/CodeLlama-7b-hf", output_dir="./code_llama_lunar"):
        """
        Initialize code generator for fine-tuning and inference.

        Args:
            model_name (str): Base model name (default: CodeLlama-7b-hf).
            output_dir (str): Directory to save fine-tuned model.
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        logger.info(colored(f"Initializing CodeGenerator with {model_name} on {self.device} 🚀", "yellow"))

    def load_model(self):
        """Load Code LLaMA model with 4-bit quantization and LoRA."""
        try:
            logger.info(colored("Loading model and tokenizer...", "cyan"))
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                load_in_4bit=True,  # Optimize for smaller GPUs
                torch_dtype=torch.float16,
                device_map="auto"
            )
            lora_config = LoraConfig(
                r=16,  # Increased rank for better adaptation
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(self.model, lora_config)
            logger.info(colored("Model loaded with LoRA! Ready to mod! 🎮", "green"))
        except Exception as e:
            logger.error(colored(f"Failed to load model: {e}", "red"))
            raise

    def preprocess_dataset(self, data_file="lunar_analyzer.jsonl"):
        """
        Load and preprocess dataset for fine-tuning.

        Args:
            data_file (str): Path to JSONL dataset.

        Returns:
            Dataset: Tokenized dataset.
        """
        if not os.path.exists(data_file):
            logger.warning(colored("Dataset not found! Generating sample data...", "yellow"))
            self.generate_sample_dataset(data_file)

        try:
            dataset = load_dataset("json", data_files=data_file)
            if "train" not in dataset:
                logger.error(colored("Dataset must have a 'train' split!", "red"))
                raise ValueError("Invalid dataset split")
            
            def tokenize_function(examples):
                return self.tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
            
            tokenized_dataset = dataset.map(tokenize_function, batched=True)
            logger.info(colored(f"Dataset loaded and tokenized: {len(dataset['train'])} examples", "cyan"))
            return tokenized_dataset
        except Exception as e:
            logger.error(colored(f"Dataset preprocessing failed: {e}", "red"))
            raise

    def generate_sample_dataset(self, output_file="lunar_analyzer.jsonl"):
        """Generate a sample dataset based on LunarAnalyzer code snippets."""
        sample_snippets = [
            {"text": "def toggle_auto_aim(self):\n    self.auto_aim = not self.auto_aim\n    logger.info(f\"Auto-aim {'enabled' if self.auto_aim else 'disabled'}\")"},
            {"text": "def capture_snapshot(self, annotate=True):\n    frame = np.array(self.screen.grab(detection_box))\n    cv2.imwrite(filename, frame)"},
            {"text": "def draw_esp(self, detections):\n    for det in detections:\n        x, y, w, h, conf = det\n        self.overlay_canvas.create_rectangle(x, y, x + w, y + h, outline=color)"},
            {"text": "def adjust_fov(self, game_fov):\n    self.fov_scale = 90 / max(1, game_fov)\n    self.box_constant = int(320 * self.fov_scale)"}
        ]
        try:
            with open(output_file, "w") as f:
                for snippet in sample_snippets:
                    json.dump(snippet, f)
                    f.write("\n")
            logger.info(colored(f"Sample dataset generated: {output_file}", "green"))
        except Exception as e:
            logger.error(colored(f"Failed to generate sample dataset: {e}", "red"))
            raise

    def fine_tune(self, data_file="lunar_analyzer.jsonl"):
        """Fine-tune the model on the provided dataset."""
        try:
            self.load_model()
            dataset = self.preprocess_dataset(data_file)
            
            training_args = TrainingArguments(
                output_dir=self.output_dir,
                per_device_train_batch_size=2,
                num_train_epochs=3,
                learning_rate=2e-4,  # Optimized for LoRA
                fp16=True,
                gradient_accumulation_steps=4,  # For smaller GPUs
                warmup_steps=100,
                logging_steps=10,
                save_strategy="epoch",
                evaluation_strategy="no",  # Add validation if dataset supports it
                report_to="none",
                logging_dir=f"{self.output_dir}/logs"
            )
            
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset["train"],
                tokenizer=self.tokenizer
            )
            
            logger.info(colored("Starting fine-tuning... Let's make some modding magic! ✨", "yellow"))
            trainer.train()
            self.model.save_pretrained(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)
            logger.info(colored(f"Model saved to {self.output_dir} 🎉", "green"))
        except Exception as e:
            logger.error(colored(f"Fine-tuning failed: {e}", "red"))
            raise

    def generate_code(self, prompt, max_length=200):
        """
        Generate code using the fine-tuned model.

        Args:
            prompt (str): Input prompt for code generation.
            max_length (int): Maximum length of generated code.

        Returns:
            str: Generated code snippet.
        """
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                num_return_sequences=1,
                do_sample=True,
                top_p=0.9,
                temperature=0.7
            )
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(colored(f"Generated code: {generated}", "cyan"))
            return generated
        except Exception as e:
            logger.error(colored(f"Code generation failed: {e}", "red"))
            return None

    def integrate_with_gui(self, gui):
        """
        Integrate code generation with LunarAnalyzerGUI.

        Args:
            gui: Instance of LunarAnalyzerGUI.
        """
        try:
            frame = tk.Frame(gui.utilities_tab, bg='#2e2e2e')
            frame.pack(fill='both', expand=True, padx=10, pady=10)

            tk.Label(frame, text="Code Generator", bg='#2e2e2e', fg='white', font=("Arial", 12, "bold")).pack(pady=5)
            tk.Label(frame, text="Enter Prompt:", bg='#2e2e2e', fg='white').pack()
            prompt_entry = tk.Entry(frame, bg='#1e1e1e', fg='white', width=40)
            prompt_entry.pack()
            output_text = tk.Text(frame, height=5, bg='#1e1e1e', fg='white', wrap=tk.WORD)
            output_text.pack(pady=5)

            def generate():
                prompt = prompt_entry.get()
                if prompt:
                    code = self.generate_code(prompt)
                    output_text.delete(1.0, tk.END)
                    output_text.insert(tk.END, code or "Generation failed!")
                    messagebox.showinfo("Code Generated", "Check the output below!")
                else:
                    messagebox.showwarning("Input Error", "Please enter a prompt!")

            tk.Button(frame, text="Generate Code", command=generate).pack(pady=5)
            logger.info(colored("Code generator integrated into GUI!", "green"))
        except Exception as e:
            logger.error(colored(f"GUI integration failed: {e}", "red"))
            messagebox.showerror("Error", f"GUI integration failed: {e}")

# Update LunarAnalyzerGUI to include code generation
def update_lunar_analyzer_gui():
    """Update LunarAnalyzerGUI to include code generation utility."""
    original_setup = LunarAnalyzerGUI.setup_utilities_tab

    def new_setup_utilities_tab(self):
        original_setup(self)
        code_generator = CodeGenerator()
        code_generator.integrate_with_gui(self)

    LunarAnalyzerGUI.setup_utilities_tab = new_setup_utilities_tab

def main():
    """Main function to run fine-tuning and start GUI."""
    os.system('cls' if os.name == 'nt' else 'clear')
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
    print(colored('''
    | |
    | |    _   _ _ __   __ _ _ __
    | |   | | | | '_ \ / _` | '__|
    | |___| |_| | | | | (_| | |
    \_____/\__,_|_| |_|\__,_|_|
    (Neural Network Analyzer)''', "yellow"))

    args = parse_arguments()
    
    config_path = "lib/config/config.json"
    if args.setup or not os.path.exists(config_path):
        logger.warning(colored("Sensitivity configuration not found", "yellow"))
        setup_sensitivity()
    
    os.makedirs("lib/data", exist_ok=True)

    # Fine-tune model if requested
    if args.fine_tune:
        code_generator = CodeGenerator()
        code_generator.fine_tune("lunar_analyzer.jsonl")

    # Update GUI with code generator
    update_lunar_analyzer_gui()

    analyzer = LunarAnalyzerGUI(
        box_constant=args.box_constant,
        collect_data=args.collect_data,
        debug=args.debug,
        plot_interval=args.plot_interval
    )
    
    listener = keyboard.Listener(on_release=lambda key: on_release(key, analyzer))
    listener.start()
    
    try:
        analyzer.start()
    except KeyboardInterrupt:
        analyzer.clean_up()
    except Exception as e:
        logger.critical(colored(f"Unexpected error: {e}", "red"))
        analyzer.clean_up()

def parse_arguments():
    """Parse command-line arguments with fine-tuning option."""
    parser = argparse.ArgumentParser(description="LunarAnalyzer: Neural Network-Based Game Analyzer")
    parser.add_argument('--collect_data', action='store_true', help="Enable data collection mode")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode")
    parser.add_argument('--setup', action='store_true', help="Force sensitivity setup")
    parser.add_argument('--fine_tune', action='store_true', help="Fine-tune Code LLaMA model")
    parser.add_argument('--box_constant', type=int, default=320, help="Detection box constant (default: 320)")
    parser.add_argument('--plot_interval', type=int, default=100, help="Number of logs to plot (default: 100)")
    return parser.parse_args()

if __name__ == "__main__":
    main()