from huggingface_hub import InferenceClient
from config import BASE_MODEL, MY_MODEL, HF_TOKEN
from transformers import pipeline


class SchoolChatbot:
    """
    This class is extra scaffolding around a model. Modify this class to specify how the model recieves prompts and generates responses.

    Example usage:
        chatbot = SchoolChatbot()
        response = chatbot.get_response("What schools offer Spanish programs?")
    """

    def __init__(self):
        """
        Initialize the chatbot with a HF model ID
        """
        model_id = MY_MODEL if MY_MODEL else BASE_MODEL # define MY_MODEL in config.py if you create a new model in the HuggingFace Hub
        self.client = InferenceClient(model=model_id, token=HF_TOKEN)
        self.pipeline = pipeline(
            "text-generation",
            model=model_id,
            device_map="auto"  # works in GPU Spaces; safe to use on CPU too
        )

        # self.pipeline = pipeline("text2text-generation", model=model_id)
        
    def format_prompt_tokenizer(self, message, history):
        """
        TODO: Implement this method to format the user's input into a proper prompt.
        
        This method should:
        1. Add any necessary system context or instructions
        2. Format the user's input appropriately
        3. Add any special tokens or formatting the model expects

        Args:
            user_input (str): The user's question about Boston schools

        Returns:
            str: A formatted prompt ready for the model
        
        Example prompt format:
            "You are a helpful assistant that specializes in Boston schools...
             User: {user_input}
             Assistant:"
        """
        system_prompt = """
You are a helpful assistant that specializes in Boston schools. You should always provide accurate information, based ONLY on the SCHOOL RULES document below. 

If an information is not in the SCHOOL RULES document, you should respond with "Sorry, I can't help with that." Only output the answer to the user's immediate question, nothing else. Do not try to predict the next user question.

SCHOOL RULES:

## **Kindergarten School Eligibility**

* Eligible for all.   
* K2 is full-day kindergarten and is offered in all BPS elementary schools.  
* Assignment to K2 is guaranteed for all applicants, but not guaranteed for a specific school.  
* Assignment to K0 and K1 is limited and not guaranteed due to space constraints.  
* Massachusetts law requires children to attend school starting the year they turn 6\.
             """
        messages = [
            {
            "role": "system",
            "content": system_prompt,
        }
        ]

        for user, assistant in history:
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": assistant})

        # Add the current message
        messages.append({"role": "user", "content": message})

        # Format it using the tokenizer's chat template
        prompt = self.pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True  # adds Assistant: at the end
        )

        return prompt
    
    def format_prompt(self, message, history):
        prompt = """
        You are a helpful assistant that specializes in Boston schools. You should always provide accurate information, based ONLY on the SCHOOL RULES document below. 
If an information is not in the SCHOOL RULES document, you should respond with "Sorry, I can't help with that." Only output the answer to the user's immediate question, nothing else. Do not try to predict the next user question.

SCHOOL RULES:

## **Kindergarten School Eligibility**

* Eligible for all.   
* K2 is full-day kindergarten and is offered in all BPS elementary schools.  
* Assignment to K2 is guaranteed for all applicants, but not guaranteed for a specific school.  
* Assignment to K0 and K1 is limited and not guaranteed due to space constraints.  
* Massachusetts law requires children to attend school starting the year they turn 6. \n
            """
        for user, assistant in history:
            prompt += f"User: {user}\nAssistant: {assistant}\n"
        prompt += f"User: {message}\nAssistant:"
        
        return prompt
        
                
    def get_response(self, message, history):
        """
        TODO: Implement this method to generate responses to user questions.
        
        This method should:
        1. Use format_prompt() to prepare the input
        2. Generate a response using the model
        3. Clean up and return the response

        Args:
            user_input (str): The user's question about Boston schools

        Returns:
            str: The chatbot's response

        Implementation tips:
        - Use self.format_prompt() to format the user's input
        - Use self.client to generate responses
        """
    
        self.prompt = self.format_prompt_tokenizer(message, history)

        output = self.pipeline(self.prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        generated_text = output[0]["generated_text"]
        new_text = generated_text[len(self.prompt):]
        return new_text.strip()  # Remove leading/trailing whitespace