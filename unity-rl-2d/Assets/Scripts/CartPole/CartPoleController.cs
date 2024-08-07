// REF: https://www.gymlibrary.dev/environments/classic_control/cart_pole/

using UnityEngine;
using System.Collections;

namespace CartPole
{
    public class CartPoleController : AgentController
    {
        // ゲームオブジェクトのRigidbody2Dの参照
        private Rigidbody2D cartRigidbody;
        private Rigidbody2D poleRigidbody;

        // 送信するデータ
        private State state;
        private int done;
        private float reward;

        // ゲームの設定

        public float moveStep = 0.02f; // 移動のステップ量(大きすぎるとうまくいかない)
    
        private int stepCount = 0; 
        private int maxSteps = 500; // stepCountがこの値に達したらゲーム終了
    
        public float maxPoleAngle = 12f; // ゲーム終了とする最大角度（度）
        public float maxCartPosition = 2.4f; // ゲーム終了とする最大位置

        public override void Setup()
        {
            cartRigidbody = GetComponent<Rigidbody2D>();
            poleRigidbody =  GetComponent<HingeJoint2D>().connectedBody;
            ResetGame();
        }

        public override void Action(string data)
        {
            ReceiveData receiveData = JsonUtility.FromJson<ReceiveData>(data);
            Move(receiveData.action);
            stepCount++;
        }

        // 行動 -> シミュレーション時間の更新した後に送信する想定
        public override string BuildSendData(int worldTime)
        {
            // State
            GetState();
            // 報酬, ゲーム終了フラグ
            CheckGameStatus();
    
            SendData sendData = new SendData
            {
                state = state,
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
    
        private void Move(int action)
        {
            Vector2 position = cartRigidbody.position;
    
            if (action == 0)
            {
                // 左に移動
                position.x -= moveStep;
            }
            else if (action == 1)
            {
                // 右に移動
                position.x += moveStep;
            }
            // Rigidbody2Dの位置を設定
            cartRigidbody.position = position;
        }
    
        private void CheckGameStatus()
        {
            // ポールの回転角度
            float poleAngle = poleRigidbody.rotation;
            // カートの位置を取得
            Vector2 cartPosition = cartRigidbody.position;

            if (Mathf.Abs(poleAngle) > maxPoleAngle)
            {
                Debug.Log("Failed: Pole angle exceeded maximum limit!");
                Debug.Log("stepCount: " + stepCount);
                reward = 0;
                done = 1;
            }
            else if (Mathf.Abs(cartPosition.x) > maxCartPosition)
            {
                Debug.Log("Failed: Cart position exceeded maximum limit!");
                Debug.Log("stepCount: " + stepCount);
                reward = 0;
                done = 1;
            }
            else if (stepCount > maxSteps)
            {
                Debug.Log("Success: Maximum steps reached!");
                Debug.Log("stepCount: " + stepCount);
                reward = 1;
                done = 1;
            }
            else
            {
                reward = 1;
                done = 0;
            }
        }

        private void GetState()
        {
            state = new State
            {
                cartPosition = cartRigidbody.position,
                cartVelocity = cartRigidbody.velocity,
                poleAngle = poleRigidbody.rotation,
                poleAngularVelocity = poleRigidbody.angularVelocity
            };
        }
    
        public override void ResetGame()
        {
            // カートのリセット
            cartRigidbody.velocity = Vector2.zero;
            cartRigidbody.position = Vector2.zero;
    
            // ポールのリセット
            poleRigidbody.position = new Vector2(0f, 2.4f);
            poleRigidbody.angularVelocity = 0f;
            float poleRotation = Random.Range(-0.05f, 0.05f);
            poleRigidbody.rotation = poleRotation;
            poleRigidbody.transform.rotation = Quaternion.Euler(0, 0, poleRotation);
    
            // ステップ数のリセット
            stepCount = 0;

            // 送信データのリセット
            GetState();
            reward = 0f;
            done = 0;
        }
    }

    public class ReceiveData
    {
        public int action;
        public int worldTime;

    }

    [System.Serializable]
    public class State {
        public Vector2 cartPosition;
        public Vector2 cartVelocity;
        public float poleAngle;
        public float poleAngularVelocity;
    }

    [System.Serializable]
    public class SendData
    {
        public State state;
        public float reward;
        public int done;
        public int worldTime;
    }
}
