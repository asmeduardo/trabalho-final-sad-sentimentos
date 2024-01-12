import os
import tempfile
from flask import Flask, render_template, request, jsonify
from classifier import Classifier

app = Flask(__name__)
classifier = Classifier(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/treinar', methods=['POST'])
def treinar_modelo():
    try:
        file = request.files['file']
        classifier.treinar_modelo(file)
        return jsonify({'message': 'Treinamento concluído com sucesso!', 'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'})

@app.route('/classificar', methods=['POST'])
def classificar_comentarios():
    try:
        file = request.files['file']

        # Crie um diretório temporário
        temp_dir = tempfile.mkdtemp()

        # Salve o arquivo no diretório temporário
        temp_file_path = os.path.join(temp_dir, 'comentarios_nao_classificados.xlsx')
        file.save(temp_file_path)

        result_html, img_path = classifier.classificar_comentarios(temp_file_path)

        # Remover o arquivo de comentários não classificados após processá-lo
        os.remove(temp_file_path)

        return render_template('results.html', result=result_html, img_path=img_path)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
