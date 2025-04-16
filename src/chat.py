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
        # self.pipeline = pipeline(
        #     "text-generation",
        #     model=model_id,
        #     device_map="auto"  # works in GPU Spaces; safe to use on CPU too
        # )
        self.system_prompt = """
<|system|> You are a helpful assistant that specializes in Boston schools. You should always provide accurate information, based ONLY on the SCHOOL RULES document below. 

If an information is not in the SCHOOL RULES document, you should respond with "Sorry, I can't help with that." Only output the answer to the user's immediate question, nothing else. Do not try to predict the next user question.

SCHOOL RULES:

## **Kindergarten School Eligibility**

## **Kindergarten School Eligibility**

* Eligible for all.   
* K2 is full-day kindergarten and is offered in all BPS elementary schools.  
* Assignment to K2 is guaranteed for all applicants, but not guaranteed for a specific school.  
* Assignment to K0 and K1 is limited and not guaranteed due to space constraints.  
* Massachusetts law requires children to attend school starting the year they turn 6\.

## **Elementary School Eligibility (Grades 1-6)**

* Eligibility based on school assignment rules (see below)  
* Most BPS elementary schools serve grades K2 through Grade 6\.  
* Some schools offer early education options starting at K0 or K1.  
* Rising Grade 7 students in a K-6 school must submit a school choice form to select a new school.

## **Middle School Eligibility (Grades 6-8)**

* Eligibility based on school assignment rules (see below)

## **School Assignment Rules (in detail) (K2 to Grade 8\)**

* BPS uses a Home-Based Assignment Plan:  
  * School assignment is based on a **student’s home address**.  
  * Each family receives a **customized list** of eligible schools.  
    * Includes **all schools within 1 mile** of the home.  
    * May also include **nearby high-quality schools**, based on BPS's **School Quality Framework (SQF)**.  
  * Additional schools, called **Option Schools**, may be added to ensure every student is offered a seat and programmatic variety.  
  * Families can also choose any **citywide school** and, in some cases, have **regional options**.  
  * Most families will see **10 to 14 school options**.  
  * An **algorithm (similar to a lottery)** is used for assignments; top choices are **not guaranteed**.  
  * **K0 and K1 assignments are not guaranteed** due to limited seats.  
  * **Best chance of assignment** to preferred schools is for students registering in **January** for Grades **K0, K1, K2, 6, 7, or 9**.  
* Age restrictions may apply, consult age chart on the website.

## **High School Eligibility (Grades 7-12)**

* All BPS high schools are citywide — every student living in Boston is eligible to apply to all of them.  
* Most high schools serve Grades 7-12.  
* Some high schools have special admission requirements.

## **Registration**

* **Age-based eligibility and required documents**: Children must meet the age cutoffs for each grade by **September 1, 2025**, and families must provide key documents including proof of age, immunizations, physical exam, parent ID, and **two separate proofs of Boston residency**.  
* **Registration is not complete until you meet with a BPS registration specialist**: Families can **pre-register online** to save time, but **final enrollment requires an appointment** (virtual or in-person) with a BPS Welcome Center representative.  
* Priority registration for students entering K0, K1, K2, 6, 7, or 9 runs from January 6 to February 7, 2025, with assignment notifications on March 31, 2025\.  
* Registration for all other grades runs from February 10 to April 4, 2025, with assignment notifications on May 31, 2025\.

## **Required Documents for Registration**

Families must provide:

* Child’s Birth Certificate, Passport, or I-94  
* Child’s Immunization Record  
* Child’s Physical Exam Record (within 1 year)  
* Parent/Guardian Photo ID  
* Two proofs of Boston residency from different categories (e.g., utility bill, lease, bank statement)

## **Contact**

* call the BPS Welcome Center at 617-635-9010  
* Watch prev information session: [https://k12-bostonpublicschools.zoom.us/rec/play/ZhU7qLQrhF3WRvr0OYfbr-G3Kadhiw8VpLaO7nQy6X29k8i-SnYE4yDDiKrWKuZUXTu13FxUKb85mDU2.UCpgyLDazDkHuhYw](https://k12-bostonpublicschools.zoom.us/rec/play/ZhU7qLQrhF3WRvr0OYfbr-G3Kadhiw8VpLaO7nQy6X29k8i-SnYE4yDDiKrWKuZUXTu13FxUKb85mDU2.UCpgyLDazDkHuhYw)  
* learn more about school choice: [https://boston.explore.avela.org/](https://boston.explore.avela.org/)  
* visit: [https://www.bostonpublicschools.org/](https://www.bostonpublicschools.org/)

             """

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
        messages = [
            {
            "role": "system",
            "content": self.system_prompt,
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
        prompt = self.system_prompt
        for user, assistant in history:
            prompt += f"<|user|> {user}\n<|assistant|> {assistant}\n"
        prompt += f"<|user|> {message}\n<|assistant|>"
        
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
    
        self.prompt = self.format_prompt(message, history)

        # output = self.pipeline(self.prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)


        
        try:
            print("Generating response...")
            output = self.client.text_generation(
                self.prompt,
                max_new_tokens=300,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                return_full_text=False
            )
            generated_text = output #.strip[0]["generated_text"]
            new_text = generated_text[len(self.prompt):]
            return generated_text.strip()  # Remove leading/trailing whitespace
            
        except Exception as e:
            print(f"API error: {e}")
            return f"I apologize, but I encountered an error: {str(e)}"

        return new_text.strip()  # Remove leading/trailing whitespace