error_prompt = """
Fix the following code by adding missing cv2 import:
cv2.imwrite(frame_filename, frame)
"""
inputs = tokenizer(error_prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_length=300, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))