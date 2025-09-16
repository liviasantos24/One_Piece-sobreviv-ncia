from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

modelo_pipeline = joblib.load('onepiece_survival_model.pkl')

@app.route('/')
def home():
    
    return render_template('op.html')

@app.route('/prever', methods=['POST'])
def prever():
    try:
       
        forca = int(request.form['forca'])
        inteligencia = int(request.form['inteligencia'])
        velocidade = int(request.form['velocidade'])
        tipo_fruta = request.form['tipo_fruta']
        habilidade = request.form['habilidade']
        nome = request.form['nome']

        dados_entrada = pd.DataFrame([[forca, inteligencia, velocidade, tipo_fruta, habilidade]], 
                                     columns=['Forca', 'Inteligencia', 'Velocidade', 'Tipo_Fruta', 'Habilidade'])
        

        previsao_binaria = modelo_pipeline.predict(dados_entrada)
        
        if previsao_binaria[0] == 1:
            resultado_texto = f"{nome} tem uma alta chance de sobreviver!"
        else:
            resultado_texto = f"{nome} pode não sobreviver..."
        
        return render_template('resultado.html', resultado=resultado_texto, nome=nome)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

