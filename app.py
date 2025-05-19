import streamlit as st
from openai import OpenAI
import json
import pandas as pd
import time
import plotly.express as px

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Initialize session state variables
if "quiz_started" not in st.session_state:
    st.session_state.quiz_started = False
if "questions" not in st.session_state:
    st.session_state.questions = []
if "answers" not in st.session_state:
    st.session_state.answers = []
if "current_index" not in st.session_state:
    st.session_state.current_index = 0
if "score" not in st.session_state:
    st.session_state.score = 0
if "quiz_start_time" not in st.session_state:
    st.session_state.quiz_start_time = None
if "confidence_ratings" not in st.session_state:
    st.session_state.confidence_ratings = []

st.title("NBME-Style Custom Quiz Generator")

# Quiz setup
if not st.session_state.quiz_started:
    topic = st.text_input("Enter a medical topic (e.g., nephrotic syndrome)")
    difficulty = st.selectbox("Select difficulty", ["Easy", "Moderate", "Hard"])
    step_level = st.selectbox("Select USMLE Step", ["1", "2 CK", "3"])
    num_questions = st.slider("Select number of questions", 5, 20, 10)

    if st.button("Start Quiz") and topic:
        with st.spinner(f"Generating {num_questions} questions... (This may take a few moments)"):
            try:
                prompt = f"""
                Generate {num_questions} USMLE Step {step_level}-style multiple-choice questions about {topic}, formatted as a single JSON object with numbered keys.
                Each value should be an object with the following keys:
                - question
                - options (list of 5 answer choices labeled A–E)
                - correct_answer (e.g., \"A\")
                - explanation

                The topic is: {topic}
                The difficulty level is: {difficulty}
                Output only the JSON object. Do not include any prose or headers.
                """

                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )
                content = response.choices[0].message.content.strip()
                questions_obj = json.loads(content)
                st.session_state.questions = list(questions_obj.values())
                st.session_state.quiz_started = True
                st.session_state.quiz_start_time = time.time()
            except Exception as e:
                st.error(f"Failed to generate quiz: {e}")

# Quiz in progress
if st.session_state.quiz_started and st.session_state.current_index < len(st.session_state.questions):
    total_questions = len(st.session_state.questions)
    st.progress(st.session_state.current_index / total_questions)

    q = st.session_state.questions[st.session_state.current_index]
    st.subheader(f"Question {st.session_state.current_index + 1} of {total_questions}")
    st.write(q["question"])

    options_dict = {}
    for opt in q["options"]:
        if len(opt) >= 3 and opt[1] == ".":
            letter = opt[0]
            label = opt[3:].strip()
        else:
            letter = opt[0]
            label = opt[1:].strip()
        options_dict[letter] = label

    choice = st.radio(
        "Choose one answer:",
        list(options_dict.items()),
        format_func=lambda x: f"{x[0]}. {x[1]}",
        key=f"question_{st.session_state.current_index}"
    )

    confidence = st.radio(
        "How confident were you in your answer?",
        ["Very", "Somewhat", "Not at all"],
        key=f"confidence_{st.session_state.current_index}"
    )

    if st.button("Submit Answer"):
        st.session_state.answers.append({
            "user_answer": choice[0],
            "correct_answer": q["correct_answer"],
            "explanation": q["explanation"],
            "question": q["question"],
            "options": options_dict
        })
        st.session_state.confidence_ratings.append(confidence)

        if choice[0] == q["correct_answer"]:
            st.session_state.score += 1

        st.session_state.current_index += 1
        st.rerun()

# Quiz complete
if st.session_state.current_index >= len(st.session_state.questions):
    st.success(f"Quiz complete! You scored {st.session_state.score}/{len(st.session_state.questions)}.")

    elapsed_time = time.time() - st.session_state.quiz_start_time if st.session_state.quiz_start_time else 0
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    st.markdown(f"⏱️ Total time: {minutes} min {seconds} sec")

    st.subheader("Review Your Answers")
    for i, answer in enumerate(st.session_state.answers):
        with st.expander(f"Question {i + 1} Review"):
            st.markdown(f"**{answer['question']}**")
            for key, val in answer["options"].items():
                st.markdown(f"- **{key}**: {val}")
            if answer["user_answer"] == answer["correct_answer"]:
                st.success(f"Your answer: {answer['user_answer']} (Correct)")
            else:
                st.error(f"Your answer: {answer['user_answer']} (Incorrect). Correct: {answer['correct_answer']}")
            st.markdown(f"_Explanation_: {answer['explanation']}")
            st.markdown(f"_Confidence_: {st.session_state.confidence_ratings[i]}")

    missed_explanations = [a['explanation'] for a in st.session_state.answers if a['user_answer'] != a['correct_answer']]
    if missed_explanations:
        joined_explanations = "\n\n".join(missed_explanations)
        feedback_prompt = f"Based on these explanations from incorrect answers, suggest 2-3 key medical concepts or systems the student should focus on:\n\n{joined_explanations}"
        try:
            feedback_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": feedback_prompt}],
            temperature=0.3
        )
            suggestion = feedback_response.choices[0].message.content
            st.subheader("📚 Suggested Focus Areas")
            st.markdown(suggestion, unsafe_allow_html=True)
        
        except Exception as e:
            st.warning(f"Unable to generate study suggestions: {e}")

    # Per-topic mastery breakdown
    st.subheader("🧩 Per-Topic Mastery Breakdown")
    topic_summary_prompt = """Extract the key concept or dominant topic tested in each of the following questions, then group by topic and count how many the user got correct vs incorrect. Return a JSON object like this: {"Biochemistry": {"Correct": 2, "Incorrect": 1}, ...}

"""
    for i, a in enumerate(st.session_state.answers):
        correct = a['user_answer'] == a['correct_answer']
        topic_summary_prompt += f"Q{i+1}: {a['question']}\nAnswer: {a['user_answer']}\nCorrect Answer: {a['correct_answer']}\nExplanation: {a['explanation']}\nResult: {'Correct' if correct else 'Incorrect'}\n\n"

    try:
        topic_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": topic_summary_prompt}],
            temperature=0.3
        )
        topic_data = json.loads(topic_response.choices[0].message.content.strip())
        topic_df = pd.DataFrame([{"Topic": k, "Correct": v.get("Correct", 0), "Incorrect": v.get("Incorrect", 0)} for k, v in topic_data.items()])
        topic_df["Total"] = topic_df["Correct"] + topic_df["Incorrect"]
        topic_df["Mastery"] = topic_df["Correct"] / topic_df["Total"] * 100
        fig_topic = px.bar(topic_df, x="Topic", y="Mastery", color="Mastery",
                           title="Mastery by Topic (%)", color_continuous_scale="Blues")
        st.plotly_chart(fig_topic)
    except Exception as e:
        st.warning(f"Unable to generate topic breakdown: {e}")

    # Performance chart
    st.subheader("📊 Performance Summary")
    correct = st.session_state.score
    incorrect = len(st.session_state.answers) - correct
    df_summary = pd.DataFrame({"Result": ["Correct", "Incorrect"], "Count": [correct, incorrect]})
    fig = px.pie(df_summary, values="Count", names="Result", color="Result",
                 color_discrete_map={"Correct": "green", "Incorrect": "red"},
                 title="Quiz Performance", hole=0.4)
    st.plotly_chart(fig)

    # Confidence vs Correctness Bar Chart
    st.subheader("🧠 Confidence vs Accuracy")
    df_conf = pd.DataFrame({
        "Confidence": st.session_state.confidence_ratings,
        "Correct": [a['user_answer'] == a['correct_answer'] for a in st.session_state.answers]
    })
    df_conf_summary = df_conf.groupby(["Confidence", "Correct"]).size().reset_index(name="Count")
    fig_conf = px.bar(df_conf_summary, x="Confidence", y="Count", color="Correct",
                      barmode="group", title="Confidence vs Accuracy",
                      color_discrete_map={True: "green", False: "red"})
    st.plotly_chart(fig_conf)

    # Export to CSV
    result_df = pd.DataFrame(st.session_state.answers)
    result_df["Confidence"] = st.session_state.confidence_ratings
    csv_export = result_df.to_csv(index=False)
    if st.download_button("Download Results as CSV", csv_export, file_name="quiz_results.csv"):
        st.success("Download started")

    # Export topic summary
    if 'topic_df' in locals():
        topic_csv = topic_df.to_csv(index=False)
        if st.download_button("Download Topic Breakdown as CSV", topic_csv, file_name="topic_breakdown.csv"):
            st.success("Topic breakdown downloaded")
        st.success("Download started")

    # Reset quiz button
    if st.button("Take Another Quiz"):
        for key in ["quiz_started", "questions", "answers", "current_index", "score", "quiz_start_time", "confidence_ratings"]:
            del st.session_state[key]
        st.rerun()

