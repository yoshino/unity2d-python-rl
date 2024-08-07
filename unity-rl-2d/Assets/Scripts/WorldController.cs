using UnityEngine;
using System.Collections;

public class WorldController : MonoBehaviour
{
    public AgentController agentController;
    public UDPCommunication udpCommunication;
    private bool isSimulationPaused = false;
    private float pauseSeconds = 5.0f;

    // server側と同期するために利用する
    private int worldTime = 0;

    // トレーニング中はframeCountThreshold=1にする
    private int frameCount = 0;
    private int frameCountThreshold = 1;

    void Start()
    {
        // 物理シミュレーション時間を手動で進める
        Physics2D.simulationMode = SimulationMode2D.Script;

        // UDP通信の設定
        udpCommunication.Setup();

        // Agentの初期設定
        agentController.Setup();

        // バックグラウンドで実行を続ける設定
        Application.runInBackground = true;

        // 初期データの送信
        SendData();
    }

    void Update()
    {
        if (isSimulationPaused) {
            return;
        }

        frameCount++;

        if (udpCommunication.IsAvailable() && frameCount % frameCountThreshold == 0)
        {
            string data = udpCommunication.ReceiveData();
            
            if(!agentController.ValidateReceiveData(data, worldTime))
            {
                return;
            }

            // Agentの行動
            agentController.Action(data);

            // worldTimeの更新
            worldTime++;

            // Time.fixedDeltaTime: 0.02f
            Physics2D.SyncTransforms();
            Physics2D.Simulate(Time.fixedDeltaTime);

            if (agentController.IsDone())
            {
                SendData();
                
                // 終了フラグを送信してからリセット
                agentController.ResetGame();
                Physics2D.SyncTransforms();
                Physics2D.Simulate(Time.fixedDeltaTime);
                // StartCoroutine(Pause());

                SendData();
            } else {
                SendData();
            }
        }
    }

    void SendData()
    {
        string sendData = agentController.BuildSendData(worldTime);
        udpCommunication.SendData(sendData);
    }

    void OnDestroy()
    {
        udpCommunication.Close();
    }

    private IEnumerator Pause()
    {
        {
            isSimulationPaused = true;
            yield return new WaitForSeconds(pauseSeconds);
            isSimulationPaused = false;
        }
    }
}
