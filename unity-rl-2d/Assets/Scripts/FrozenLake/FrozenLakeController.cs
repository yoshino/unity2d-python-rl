using UnityEngine;

namespace FrozenLake {
    public class FrozenLakeController : AgentController
    {
        private int state;
        private float reward;
        private int done;
        
        public override void Setup()
        {
            state = GetState();
            reward = -0.04f;
            done = 0;
        }
        
        public override void Action(string data)
        {
            ReceiveData receiveData = JsonUtility.FromJson<ReceiveData>(data);
        
            Vector2 position = MovePosition(receiveData.action);
            this.transform.position = position;

            // position = ClampPosition(position);
            state = GetState();
            reward = GetReward(position);
            done = CheckDone(position);
        }

        
        public override string BuildSendData(int worldTime)
        {
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

        public override void ResetGame()
        {
            transform.position = new Vector2(0, 3);
            state = GetState();
            reward = -0.04f;
            done = 0;
        }

        public override bool IsDone()
        {
            return done == 1;
        }


        private int GetState()
        {
            return (int)(transform.position.x + transform.position.y * 4);
        } 

        private Vector2 MovePosition(int action)
        {
            Vector2 position = ConvertToPosition(state);

            switch (action)
            {
                case 0:
                    position.y += 1;
                    break;
                case 1:
                    position.x += 1;
                    break;
                case 2:
                    position.y -= 1;
                    break;
                case 3:
                    position.x -= 1;
                    break;
            }
            return position;
        }

        private Vector2 ConvertToPosition(int state)
        {
            return new Vector2(state % 4, state / 4);
        }

        
        private float GetReward(Vector2 position)
        {
            Collider2D collider = Physics2D.OverlapPoint(position);
            if (collider != null)
            {
                if (collider.CompareTag("Hole"))
                {
                    Debug.Log("Hole!!");
                    return -1.0f;
                }
                else if (collider.CompareTag("Goal"))
                {
                    Debug.Log("Goal!!");
                    return 1.0f;
                }
            }
            return -0.04f;
        }
        
        private int CheckDone(Vector2 position)
        {
            Collider2D collider = Physics2D.OverlapPoint(new Vector2(position.x, position.y));
            if (collider != null)
            {
                if (collider.CompareTag("Hole") || collider.CompareTag("Goal"))
                {
                    return 1;
                }
            }
            return 0;
        }
        
        private Vector2 ClampPosition(Vector2 position)
        {
            position.x = Mathf.Clamp(position.x, 0, 3);
            position.y = Mathf.Clamp(position.y, 0, 3);
            return position;
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
        public int state;
        public float reward;
        public int done;
        public int worldTime;
    }
}
