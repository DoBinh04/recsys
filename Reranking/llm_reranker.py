from openai import OpenAI

client = OpenAI()


class LLMReranker:

    def __init__(self, model="gpt-4o-mini"):
        self.model = model


    def build_prompt(self, user_history, candidates):

        history_text = "\n".join([f"- {item}" for item in user_history])

        items_text = ""
        for i, item in enumerate(candidates):
            items_text += f"{i+1}. Title: {item['title']} | Category: {item['category']}\n"

        prompt = f"""
User browsing history:
{history_text}

Candidate products:
{items_text}

Task:
Rank the candidate products based on how relevant they are to the user's interests.

Return only the ranked list of item numbers (most relevant first).

Example output:
3,1,2
"""

        return prompt


    def rerank(self, user_history, candidates):

        prompt = self.build_prompt(user_history, candidates)

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        result = response.choices[0].message.content.strip()

        order = [int(i) - 1 for i in result.split(",")]

        reranked_items = [candidates[i] for i in order]

        return reranked_items