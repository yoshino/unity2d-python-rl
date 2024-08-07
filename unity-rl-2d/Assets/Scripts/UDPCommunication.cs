
using UnityEngine;
using System.Net;
using System.Net.Sockets;
using System.Text;

public class UDPCommunication : MonoBehaviour
{
    private UdpClient udpClient;
    private IPEndPoint remoteEndPoint;
    private string serverIP = "127.0.0.1";
    private int serverPort = 9000;
    private int localPort = 8000;

    public void Setup()
    {
        udpClient = new UdpClient(localPort);
        remoteEndPoint = new IPEndPoint(IPAddress.Parse(serverIP), serverPort);
    }

    public string ReceiveData()
    {
        byte[] data = udpClient.Receive(ref remoteEndPoint);
        return Encoding.UTF8.GetString(data);
    }

    public void SendData(string message)
    {
        byte[] data = Encoding.UTF8.GetBytes(message);
        udpClient.Send(data, data.Length, remoteEndPoint);
    }

    public bool IsAvailable()
    {
        return udpClient.Available > 0;
    }

    public void Close()
    {
        udpClient.Close();
    }
}
