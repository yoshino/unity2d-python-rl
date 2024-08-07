using UnityEngine;

public abstract class AgentController : MonoBehaviour
{
    public abstract void Setup();

    public abstract void Action(string data);

    public abstract string BuildSendData(int worldTime);

    public abstract bool ValidateReceiveData(string data, int worldTime);


    public abstract bool IsDone();

    public abstract void ResetGame();
}
