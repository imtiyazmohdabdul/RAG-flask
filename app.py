from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import os
from werkzeug.utils import secure_filename
from rag_engine import rag, UPLOAD_FOLDER

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'md'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        
        try:
            chunks_count = rag.ingest_file(save_path, filename)
            return jsonify({
                "success": True, 
                "message": f"Successfully indexed '{filename}'", 
                "chunks": chunks_count
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
            
    return jsonify({"error": "File type not allowed (PDF, TXT, MD only)"}), 400

@app.route('/chat_stream', methods=['POST'])
def chat_stream():
    data = request.get_json()
    question = data.get('message', '').strip()
    
    if not question:
        return jsonify({"error": "Empty question"}), 400

    # 1. Retrieve
    hits = rag.retrieve(question)
    
    # 2. Return stream
    return Response(stream_with_context(rag.generate_stream(question, hits)), content_type='text/plain')

if __name__ == '__main__':
    # Find a free port or default to 5001
    port = 5001
    print(f"Starting RAG Web Interface on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=True, threaded=True)
