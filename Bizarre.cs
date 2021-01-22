using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;

public class BallAgentLogic : Agent
{

    Rigidbody rBody;
    public Transform target;

    void Start()
    {
        rBody = GetComponent<Rigidbody>();
    }

    public override void OnEpisodeBegin()
    {
        this.rBody.angularVelocity = Vector3.zero;
        this.rBody.velocity = Vector3.zero;
        this.transform.localPosition = new Vector3(2, 0.55f, -3.8f);

        target.localPosition = new Vector3(3 + Random.value * 8, Random.value * 3, 7 + Random.value * 10 - 5);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        if (target)
        {
            sensor.AddObservation(target.localPosition);
            sensor.AddObservation(this.transform.localPosition);
            sensor.AddObservation(rBody.velocity);
        }
    }

    public float speed = 20;

    public override void OnActionReceived(float[] vectorAction)
    {
        Vector3 controlSignal = Vector3.zero;
        controlSignal.x = vectorAction[0];

        if(vectorAction[1] == 2)
        {
            controlSignal.z = 1;
        }
        else
        {
            controlSignal.z = -vectorAction[1];
        }

        if(this.transform.localPosition.x < 4.5)
        {
            rBody.AddForce(controlSignal * speed);
        }

        float distanceToTarget = Vector3.Distance(this.transform.localPosition, target.localPosition);

        if(distanceToTarget < 1.42f)
        {
            SetReward(1.0f);
            EndEpisode();
        }

        if(this.transform.localPosition.y < 0)
        {
            EndEpisode();
        }
    }

    public override void Heuristic(float[] actionsOut)
    {
        actionsOut[0] = Input.GetAxis("Vertical");
        actionsOut[1] = Input.GetAxis("Horizontal");
    }
}
