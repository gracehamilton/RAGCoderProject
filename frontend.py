from flask import Flask, render_template, request, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename

from RAGService import RAGSession

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET', 'devkey')

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'data')
os.makedirs(f"{UPLOAD_FOLDER}/unparsed", exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#Variable for allowed extensions. For now, only pdf; may expand to txt later
ALLOWED_EXTENSIONS = {'pdf'}

#function for validating file types
def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Lazy RAGSession holder (initialised on first use) and cached per-config
rag_session = None
rag_config = {}
def get_rag_session(inference_model=None, embeddings_model=None, ollama_url=None):
	global rag_session, rag_config
	#desired config based on specified values or env vars (default)
	desired = {
		'inference_model': inference_model or os.getenv('ollama_inference_model'),
		'embeddings_model': embeddings_model or os.getenv('ollama_embeddings_model'),
		'ollama_url': ollama_url or os.getenv('ollama_url')
	}
	# recreate session when config changes or if previous init failed
	if isinstance(rag_session, Exception) or rag_session is None or rag_config != desired:
		try:
			rag_session = RAGSession(inference_model=desired['inference_model'], embeddings_model=desired['embeddings_model'], ollama_url=desired['ollama_url'])
			rag_config = desired
		except Exception as e:
			rag_session = e
	return rag_session

#basic routing for index, model info
@app.route('/', methods=['GET'])
def index():
	inference_models = ['llama3.2', 'gemma4:e2b']
	embedding_models = ['nomic-embed-text', 'embeddinggemma']
	vector_dbs = ['Chroma']
	return render_template('index.html', inference_models=inference_models, embedding_models=embedding_models, vector_dbs=vector_dbs)

#routing for upload
@app.route('/upload', methods=['POST'])
def upload():
	#if no file part, flash error and reroute to index
	if 'file' not in request.files:
		flash('No file part')
		return redirect(url_for('index'))
	file = request.files['file']
	#if no file selected, flash error and reroute to index
	if file.filename == '':
		flash('No selected file')
		return redirect(url_for('index'))
	#if valid file, give it a secure filename and save to upload folder. Flash success message
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
		file.save(save_path)
		flash(f'Uploaded {filename}')
		return redirect(url_for('index'))
	else:
		#if invalid file type, flash error and reroute to index
		flash('Only PDF files are allowed')
		return redirect(url_for('index'))

#route for asking questions
@app.route('/ask', methods=['POST'])
def ask():
	model = request.form.get('inference_model')
	embed = request.form.get('embedding_model')
	vdb = request.form.get('vector_db')
	question = request.form.get('question', '').strip()
	if not question:
		flash('Please enter a question')
		return redirect(url_for('index'))

	# initialize or update the RAGSession based on user-selected models
	session = get_rag_session(inference_model=model, embeddings_model=embed)
	if isinstance(session, Exception):
		return f'Failed to initialize RAGSession: {session}', 500

	try:
		# Currently RAGSession uses env vars for model names; selections are shown in UI
		answer = session.generate_rag_response(question)
	except Exception as e:
		return f'Error generating response: {e}', 500

	inference_models = ['llama3.2', 'gemma4:e2b']
	embedding_models = ['nomic-embed-text', 'embeddinggemma']
	vector_dbs = ['Chroma']
	return render_template('index.html', answer=answer, inference_models=inference_models, embedding_models=embedding_models, vector_dbs=vector_dbs)


if __name__ == '__main__':
	app.run(host='0.0.0.0', port=5000, debug=True)