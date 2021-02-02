using Syn.Bot.Oscova;
using Syn.Bot.Oscova.Attributes;
using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class Message
{
    public string text;
    public Text text_object;
    public message_type message_type;
}

public enum message_type
{
    user, bot
}

public class BotDialog : Dialog
{
    [Expression("hello bot")]
    public void hello(Context context, Result result)
    {
        result.SendResponse("hello hacker");
    }
}

public class GameManager : MonoBehaviour
{
    public OscovaBot main_bot;

    List<Message> messages = new List<Message>();

    public GameObject chat_panel, text_object;

    public InputField chat_box;

    public Color user_color, bot_color;


    void Start()
    {
        try
        {
            main_bot = new OscovaBot();

            //main_bot.ImportWorkspace(@"[directory for oryzer structure]");

            OscovaBot.Logger.LogReceived += (s, o) =>
            {
                Debug.Log($"OscovaBot : {o.Log}");
            };

            main_bot.Dialogs.Add(new BotDialog());
            main_bot.Trainer.StartTraining();

            main_bot.MainUser.ResponseReceived += (sender, evt) =>
            {
                add_message($"Bot: {evt.Response.Text}", message_type.bot);
            };
        }

        catch (Exception ex)
        {
            Debug.LogError(ex);
        }
    }

    public void add_message(string message_text, message_type message_type)
    {
        if(messages.Count > 24)
        {
            Destroy(messages[0].text_object.gameObject);
            messages.Remove(messages[0]);
        }

        var new_message = new Message { text = message_text };

        var new_text = Instantiate(text_object, chat_panel.transform);

        new_message.text_object = new_text.GetComponent<Text>();
        new_message.text_object.text = message_text;
        new_message.text_object.color = message_type == message_type.user ? user_color : bot_color;

        messages.Add(new_message);
    }

    public void send_message_to_bot()
    {
        var user_message = chat_box.text;

        if (!string.IsNullOrEmpty(user_message))
        {
            Debug.Log($"OscovaBot: [user] {user_message}");
            add_message($"user: {user_message}", message_type.user);
            var request = main_bot.MainUser.CreateRequest(user_message);
            var evaluation_result = main_bot.Evaluate(request);
            evaluation_result.Invoke();

            chat_box.Select();
            chat_box.text = "";
        }
    }

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Return))
        {
            send_message_to_bot();
        }
    }
}
