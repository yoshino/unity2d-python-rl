# Unity2D Python RL
[Pythonで学ぶ強化学習 [改訂第2版] 入門から実践まで](https://www.amazon.co.jp/dp/4065172519)を参考に、UnityとPython(PyTorch)で実装した。

## Setup python server
main.pyを編集して、利用したいenvとagentを選択してから、python serverを起動する。

### venv
mpsを利用することができる。

```bash
python3 -m venv pyenv
source pyenv/bin/activate
pip install -r  python-server/requirements.txt
cd python-server
python main.py
```

### Docker
```bash
docker compose up
```
## Setup Unity
`unity-rl-2d`ディレクトリを指定してUnityを起動する。  
python側で選択したenvに合致するsceneを選択して、シミュレーションを開始する。
