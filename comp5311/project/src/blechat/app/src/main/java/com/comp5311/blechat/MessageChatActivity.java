package com.comp5311.blechat;

import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.Toast;

import com.comp5311.blechat.adapter.ChatroomRWAdapter;
import com.comp5311.blechat.nearby.ConnectionsActivity;
import com.google.android.gms.nearby.connection.Payload;

import java.util.ArrayList;

public class MessageChatActivity extends AppCompatActivity implements BLEChat.BLEChatMessageRoomHandler {

    BLEChat bleChat;
    RecyclerView rcChatroomMes;
    EditText etChatroomMesInput;
    ImageView btnChatroomBack;
    ImageView btnChatroomSend;

    private String targetEndpointId;
    private ChatroomRWAdapter adapter;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_message_chat);
        targetEndpointId = getIntent().getStringExtra("endpointId");
        bleChat = BLEChat.getInstance();
        bleChat.setMessageRoomClient(this);

        rcChatroomMes = findViewById(R.id.rcChatroomMes);
        etChatroomMesInput = findViewById(R.id.etChatroomMesInput);
        btnChatroomBack = findViewById(R.id.btnChatroomBack);
        btnChatroomBack.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                finish();
            }
        });

        btnChatroomSend = findViewById(R.id.btnChatroomSend);
        btnChatroomSend.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                String inputMes = etChatroomMesInput.getText().toString();
                if(inputMes.length() != 0){
                    BLEChat.Message message = bleChat.createMessage(targetEndpointId, inputMes);
                    appendMessage(message);
                    bleChat.send(message);
                }
            }
        });
        initRCChatMessage();
    }

    private void initRCChatMessage(){
        ArrayList<BLEChat.Message> messages = bleChat.getMessages(targetEndpointId);
        rcChatroomMes.setLayoutManager(new LinearLayoutManager(this));
        adapter = new ChatroomRWAdapter(messages);
        rcChatroomMes.setAdapter(adapter);
    }

    private void appendMessage(BLEChat.Message message){
        adapter.appendMessage(message);
    }

    @Override
    public void onReceive(ConnectionsActivity.Endpoint endpoint, Payload payload) {
        adapter.appendMessage(bleChat.createMessage(null, payload));
        rcChatroomMes.scrollToPosition(adapter.getItemCount()-1);
    }

    @Override
    public void onEndpointDisconnected(ConnectionsActivity.Endpoint endpoint) {
        onEndpointLostConnection(endpoint.getId());
    }

    @Override
    public void onEndpointLostConnection(String endpoint) {
        if(endpoint == targetEndpointId)
            Toast.makeText(this, "User is disconnected", Toast.LENGTH_SHORT).show();
    }

    @Override
    protected void onStop() {
        bleChat.stopAll();
        super.onStop();
    }
}