import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

class Classifier:
    def __init__(self, app):
        self.app = app
        self.vectorizer = CountVectorizer()
        self.model = MultinomialNB()

    def treinar_modelo(self, file):
        train_data = pd.read_excel(file, engine='openpyxl')
        X_train = self.vectorizer.fit_transform(train_data['comentario'])
        y_train = train_data['classificacao']
        self.model.fit(X_train, y_train)

    def classificar_comentarios(self, file):
        comentarios_nao_classificados = pd.read_excel(file, engine='openpyxl')
        X_test = self.vectorizer.transform(comentarios_nao_classificados['comentario'])
        predictions = self.model.predict(X_test)
        comentarios_nao_classificados['classificacao'] = predictions

        # Renomeie os cabeçalhos conforme necessário
        comentarios_nao_classificados = comentarios_nao_classificados.rename(columns={'comentario': 'Comentário', 'classificacao': 'Classificação'})

        img_path = self._gerar_graficos(comentarios_nao_classificados)
        result_html = comentarios_nao_classificados.to_html(classes='table table-striped table-bordered', index=False)
        return result_html, img_path

    def _gerar_graficos(self, data):
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        data['Classificação'].value_counts().plot.pie(autopct='%1.1f%%')
        plt.title('Porcentagem de Comentários Positivos')

        plt.subplot(1, 2, 2)
        words_freq = pd.Series(' '.join(data['Comentário']).split()).value_counts()[:5]
        words_freq.plot.pie(autopct='%1.1f%%')
        plt.title('Top 5 Palavras Mais Frequentes')

        img_path = "static/image/result.png"  # Alteração no caminho da imagem
        plt.savefig(os.path.join(self.app.root_path, img_path))
        return img_path

