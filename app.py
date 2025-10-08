import os
import base64
from flask import Flask, request, render_template, redirect, url_for, send_file
import psycopg2
from psycopg2.extras import RealDictCursor
from PIL import Image
from huggingface_hub import InferenceClient
import io
import threading

client = InferenceClient(
    provider="nebius", #hyperbolic 
    api_key=os.environ["HF_TOKEN"],
)

app = Flask(__name__)

conn = psycopg2.connect(
    host="localhost",
    database="dap"
)

def init_db():
    with conn.cursor() as cur:
        cur.execute('''
            CREATE TABLE IF NOT EXISTS images (
                id SERIAL PRIMARY KEY,
                filename TEXT NOT NULL,
                data BYTEA NOT NULL,
                question TEXT,
                answer TEXT
            );
        ''')
        conn.commit()

init_db()

def process_image_question(image_id, question):
    with conn.cursor() as cur:
        cur.execute("SELECT data FROM images WHERE id = %s", (image_id,))
        result = cur.fetchone()
    if result:
        img_bytes = io.BytesIO(result[0])
        img = Image.open(img_bytes).convert("RGB")
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_bytes = buffer.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        stream = client.chat.completions.create(
            model="Qwen/Qwen2.5-VL-72B-Instruct",
            messages=[
                {
                    "role": "system",
                    "content": """Ты — специалист по психологическому тесту "Рисунок человека". 
                                  Ты используешь научные методики (Венгер, Маховер, Гуденаф и др.) для анализа рисунков. 
                                  Отвечай строго в формате простого текста без Markdown."""
                },
                {
                    "role": "user",
                    "content": [
                        {   
                            "type": "text", 
                            "text": question
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.7,            # Чем ближе к 0 — тем детерминированнее
            top_p=1.0,                  # ядро выборки (top-p sampling)
            max_tokens=3072,            # Ограничение длины
            seed=42,                    # Фиксированное для воспроизводимости
            stream=True
        )
        answer = ''.join(list(chunk.choices[0].delta.content for chunk in stream))
        # Сохраняем ответ в БД
        with conn.cursor() as cur:
            cur.execute("UPDATE images SET answer = %s WHERE id = %s", (answer, image_id))
            conn.commit()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename:
            image_data = file.read()

            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO images (filename, data) VALUES (%s, %s)",
                    (file.filename, image_data)
                )
                conn.commit()

            return redirect(url_for('index'))

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT id, filename, question, answer FROM images ORDER BY id DESC")
        images = cur.fetchall()

    return render_template("index.html", images=images)

@app.route("/image/<int:image_id>")
def get_image(image_id):
    with conn.cursor() as cur:
        cur.execute("SELECT data FROM images WHERE id = %s", (image_id,))
        result = cur.fetchone()
    if result:
        return send_file(
            io.BytesIO(result[0]),
            mimetype='image/png'
        )
    else:
        return "Изображение не найдено", 404

@app.route("/ask/<int:image_id>", methods=["GET", "POST"])
def ask(image_id):
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT id, filename, question, answer FROM images WHERE id = %s", (image_id,))
        image = cur.fetchone()

    if not image:
        return "Изображение не найдено", 404

    if request.method == "POST":
        question = request.form.get("question")
        if question:
            with conn.cursor() as cur:
                cur.execute("UPDATE images SET question = %s, answer = null WHERE id = %s", (question, image_id))
                conn.commit()

            # Запускаем обработку в отдельном потоке
            thread = threading.Thread(target=process_image_question, args=(image_id, question))
            thread.start()

            return redirect(url_for('ask', image_id=image_id))

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT id, filename, question, answer FROM images WHERE id = %s", (image_id,))
        image = cur.fetchone()
    return render_template("ask.html", image=image)

    
@app.route("/delete/<int:image_id>", methods=["POST"])
def delete_image(image_id):
    with conn.cursor() as cur:
        cur.execute("DELETE FROM images WHERE id = %s", (image_id,))
        conn.commit()
    return redirect(url_for('index'))


if __name__ == "__main__":
    app.run()