using UnityEngine;

namespace Catcher
{
    public class CatcherController : AgentController
    {
        // シーンに配置されたターゲットを参照
        public GameObject targetObject; 
        
        // SendData
        private string screenShot;
        private int done = 0;
        private int reward = 0;
        
        // Other settings
        private float groundY = 0.5f;
        private Camera mainCamera;
        
        public override void Setup()
        {
            mainCamera = Camera.main;
            GetScreenShot();
            ResetGame();
        }
        
        public override void Action(string data)
        {
            ReceiveData receiveData = JsonUtility.FromJson<ReceiveData>(data);
            Move(receiveData.action);
            GetScreenShot();
            CheckDone();
        }
        
        public override string BuildSendData(int worldTime)
        
        {
            SendData sendData = new SendData
            {
                screenShot = screenShot,
                reward = reward,
                done = done,
                worldTime = worldTime
            };
            return JsonUtility.ToJson(sendData);
        }
        
        public override bool ValidateReceiveData(string data, int worldTime)
        {
            ReceiveData receiveData = JsonUtility.FromJson<ReceiveData>(data);
            return receiveData.worldTime == worldTime;
        }   

        public override bool IsDone()
        {
            return done == 1;
        }

        
        public override void ResetGame()
        {
            // ターゲットの位置をランダムにリセット
            // float spawnX = Random.Range(-4f, 4f); floatで定義すると状態数が増えるためintで定義
            int xPosition = Random.Range(-4, 4);
            Vector3 spawnPosition = new Vector3(xPosition, 10f, 0);
            targetObject.transform.position = spawnPosition;
            targetObject.GetComponent<Rigidbody2D>().velocity = Vector2.zero;
            targetObject.GetComponent<Rigidbody2D>().rotation = 0;
            reward = 0;
            done = 0;
        }
        
        void Move(int action)
        {
            Vector3 newPosition = transform.position;
        
            // actionが1の場合は何もしない(静止)
            if (action == 0)
            {
                // 左に移動
                newPosition.x -= 1.0f;
            }
            else if (action == 2)
            {
                // 右に移動
                newPosition.x += 1.0f;
            }
        
            // 移動後に位置を制限
            newPosition.x = Mathf.Clamp(newPosition.x, -4, 4);
            transform.position = newPosition;
        }
        
        void GetScreenShot()
        {
            // RenderTextureを作成
            RenderTexture renderTexture = new RenderTexture(84, 84, 24);
            mainCamera.targetTexture = renderTexture;
            Texture2D texture2D = new Texture2D(84, 84, TextureFormat.RGB24, false);
        
            // カメラの内容を読み込む
            mainCamera.Render();
            RenderTexture.active = renderTexture;
            texture2D.ReadPixels(new Rect(0, 0, renderTexture.width, renderTexture.height), 0, 0);
            texture2D.Apply();
        
            // バイト配列に変換
            byte[] imageBytes = texture2D.EncodeToJPG();
            screenShot = System.Convert.ToBase64String(imageBytes);
        
            // 後処理
            mainCamera.targetTexture = null;
            RenderTexture.active = null;
            Destroy(renderTexture);
            Destroy(texture2D);
        }
        
        void CheckDone()
        {
            // ターゲットが地面に接触したかどうかの確認
            if (targetObject != null && targetObject.transform.position.y < groundY)
            {
                Debug.Log("Failed: Target is on the ground");
                reward = -1;
                done = 1;
            // ターゲットとの距離が1未満になった場合
            } 
            else if (Vector3.Distance(transform.position, targetObject.transform.position) < 1.5f) 
            {
                Debug.Log("Success: Target is caught");
                reward = 1;
                done = 1;
            } else {
                reward = 0;
                done = 0;
            }
        }
    }
        
    public class ReceiveData
    {
        public int action;
        public int worldTime;
    }
        
    [System.Serializable]
    public class SendData
    {
        public string screenShot;
        public int done;
        public int reward;
        public int worldTime;
    }
}
