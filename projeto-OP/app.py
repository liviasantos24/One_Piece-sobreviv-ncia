from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

# Inicializa o aplicativo Flask
app = Flask(__name__)

# Carrega o modelo treinado e o pré-processador
# Certifique-se de que o caminho do arquivo está correto
modelo_pipeline = joblib.load('onepiece_survival_model.pkl')

@app.route('/')
def home():
    # Renderiza a página HTML do formulário
    return render_template('op.html')

@app.route('/prever', methods=['POST'])
def prever():
    try:
        # Extrai os dados do formulário
        forca = int(request.form['forca'])
        inteligencia = int(request.form['inteligencia'])
        velocidade = int(request.form['velocidade'])
        tipo_fruta = request.form['tipo_fruta']
        habilidade = request.form['habilidade']
        nome = request.form['nome']

        # Cria um DataFrame com os dados de entrada
        dados_entrada = pd.DataFrame([[forca, inteligencia, velocidade, tipo_fruta, habilidade]], 
                                     columns=['Forca', 'Inteligencia', 'Velocidade', 'Tipo_Fruta', 'Habilidade'])
        
        # Usa o pipeline completo para fazer a previsão
        # O pipeline já tem o pre-processamento e o modelo
        previsao_binaria = modelo_pipeline.predict(dados_entrada)
        
        # Converte a previsão binária em um resultado legível
        if previsao_binaria[0] == 1:
            resultado_texto = f"{nome} tem uma alta chance de sobreviver! 🎉"
        else:
            resultado_texto = f"{nome} pode não sobreviver... 😥"
        
        # Retorna o resultado para a página HTML
        return render_template('resultado.html', resultado=resultado_texto, nome=nome)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
