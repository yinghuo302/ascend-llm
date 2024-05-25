import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from config import InferenceConfig
from inference import LlamaInterface

def main(cli:bool,engine:LlamaInterface):
    if cli:
        while True:
            line = input()
            print(engine.predict(line))
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
        pool.submit(engine.predict,msg)
        return jsonify({"code": 200})

    @app.route("/api/getMsg", methods=["GET"])
    def getMsg():
        return jsonify(engine.getState())
    
    @app.route("/api/reset", methods=["GET"])
    def reset():
        engine.reset()
        return jsonify({"code": 200})

    app.run(
        use_reloader=False,
        host="0.0.0.0",
        port=5000
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cli', dest='cli', default=False, action='store_true',
        help="run web ui by default, if add --cli, run cli."
    )
    parser.add_argument("--kv_size", type=int, default=256)
    parser.add_argument(
        "--engine", type=str, default="acl",
        help="inference backend, onnx or acl"
    )
    parser.add_argument(
        "--sampling", type=str, default="top_k",
        help="sampling method, greedy, top_k or top_p"
    )
    parser.add_argument(
        "--sampling_value",type=float,default=10,
        help="if sampling method is seted to greedy, this argument will be ignored; if top_k, it means value of p; if top_p, it means value of p"
    )
    parser.add_argument(
        "--temperature",type=float,default=0.7,
        help="sampling temperature if sampling method is seted to greedy, this argument will be ignored."
    )
    parser.add_argument(
        "--hf-dir", type=str, default="/root/model/tiny-llama-1.1B", 
        help="path to huggingface model dir"
    )
    parser.add_argument(
        "--model", type=str, default="/root/model/tiny-llama-seq-1-key-256-int8.om", 
        help="path to onnx or om model"
    )
    args = parser.parse_args()
    cfg = InferenceConfig(
        hf_model_dir=args.hf_dir,
        model=args.model,
        max_cache_size=args.kv_size,
        sampling_method=args.sampling,
        sampling_value=args.sampling_value,
        temperature=args.temperature,
        session_type=args.engine,
    )
    engine = LlamaInterface(cfg)
    main(args.cli,engine)