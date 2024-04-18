import sys
from concurrent.futures import ThreadPoolExecutor
from config import InferenceConfig
from inference import LlamaInterface
# cfg=InferenceConfig(tokenizer="/root/model/llama-7b-chat-hf",model="/root/model/llama-seq-1-key-256-int8.om")
cfg=InferenceConfig(tokenizer="/root/model/tiny-llama-1.1B/tokenizer.model",model="/root/model/tiny-llama-seq-1-key-256-int8.om")
# cfg=InferenceConfig(tokenizer="/root/model/tiny-llama-1.1B",model="/root/model/tiny-llama-onnx-int8/llama.onnx")
infer_engine=LlamaInterface(cfg)

def inference_cli():
    while True:
        line = input()
        print(infer_engine.predict(line))
        
def main():
    if len(sys.argv) > 1:
        inference_cli()
        return
    from flask import Flask, request, jsonify
    from flask import render_template  # 引入模板插件
    from flask_cors import CORS
    pool = ThreadPoolExecutor(max_workers=2)        
    app = Flask(
        __name__,
        static_folder='./dist',  # 设置静态文件夹目录
        template_folder="./dist",
        static_url_path=""
    )

    CORS(app, resources=r'/*')
    
    @app.route('/')
    def index():
        return render_template('index.html', name='index')

    @app.route("/api/chat", methods=["POST"])
    def getChat():
        msg = request.get_json(force=True)['message']
        if len(msg) == 0:
            return jsonify({"code": 404})
        pool.submit(infer_engine.predict,msg)
        return jsonify({"code": 200})

    @app.route("/api/getMsg", methods=["GET"])
    def getMsg():
        return jsonify(infer_engine.getState())
    
    @app.route("/api/reset", methods=["GET"])
    def reset():
        infer_engine.reset()
        return jsonify({"code": 200})

    app.run(
        use_reloader=False,
        host="0.0.0.0",
        port=5000
    )

if __name__ == '__main__':
    main()