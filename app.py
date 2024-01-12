from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Criar a pasta "image" se não existir
if not os.path.exists("image"):
    os.makedirs("image")

# Dados de treinamento
train_data = pd.DataFrame(columns=['comentario', 'classificacao'])

# Modelo inicial vazio
vectorizer = CountVectorizer()
model = MultinomialNB()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/treinar', methods=['POST'])
def treinar_modelo():
    try:
        file = request.files['file']
        # Modificado para suportar arquivos XLSX
        train_data = pd.read_excel(file, engine='openpyxl')

        # Treinar o modelo
        X_train = vectorizer.fit_transform(train_data['comentario'])
        y_train = train_data['classificacao']
        model.fit(X_train, y_train)

        return jsonify({'message': 'Treinamento concluído com sucesso!'})

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/classificar', methods=['POST'])
def classificar_comentarios():
    try:
        # Receba o arquivo de comentários para classificar
        file = request.files['file']
        # Modificado para suportar arquivos XLSX
        comentarios_nao_classificados = pd.read_excel(file, engine='openpyxl')

        # Extraia os comentários
        X_test = vectorizer.transform(comentarios_nao_classificados['comentario'])

        # Faça as previsões
        predictions = model.predict(X_test)

        # Adicione as previsões aos dados originais
        comentarios_nao_classificados['classificacao'] = predictions

        # Criar gráficos
        plt.figure(figsize=(10, 5))

        # Gráfico de pizza para a porcentagem de comentários positivos
        plt.subplot(1, 2, 1)
        comentarios_nao_classificados['classificacao'].value_counts().plot.pie(autopct='%1.1f%%')
        plt.title('Porcentagem de Comentários Positivos')

        # Gráfico de pizza para as 5 palavras mais frequentes
        plt.subplot(1, 2, 2)
        words_freq = pd.Series(' '.join(comentarios_nao_classificados['comentario']).split()).value_counts()[:5]
        words_freq.plot.pie(autopct='%1.1f%%')
        plt.title('Top 5 Palavras Mais Frequentes')

        # Salvar os gráficos
        img_path = "image/result.png"
        plt.savefig(os.path.join(app.root_path, img_path))

        # Converter os dados para formato HTML
        result_html = comentarios_nao_classificados.to_html(classes='table table-striped table-bordered', index=False)

        # Retornar os resultados e caminho da imagem
        return render_template('results.html', result=result_html, img_path=img_path)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
