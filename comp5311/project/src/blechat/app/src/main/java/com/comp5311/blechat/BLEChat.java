package com.comp5311.blechat;

import android.app.Activity;

import androidx.annotation.CallSuper;
import androidx.annotation.NonNull;

import com.comp5311.blechat.nearby.ConnectionsActivity;
import com.google.android.gms.nearby.connection.ConnectionInfo;
import com.google.android.gms.nearby.connection.Payload;
import com.google.android.gms.nearby.connection.Strategy;
import com.google.android.gms.tasks.OnFailureListener;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public class BLEChat extends ConnectionsActivity{

    public interface BLEChatMessageRoomHandler {
        void onReceive(Endpoint endpoint, Payload payload);
        void onEndpointDisconnected(Endpoint endpoint);
        void onEndpointLostConnection(String endpoint);
    }

    public interface BLEChatConnectionSearchHandler {
        void onEndpointDiscovered(Endpoint endpoint);
        void onConnectionInitiated(Endpoint endpoint, ConnectionInfo connectionInfo);
        void onEndpointLostConnection(String endpoint);
    }

    public class Message {
        public String toEndpointId;
        public String message;
        public Message(String toEndpointId, String message){
            this.toEndpointId = toEndpointId;
            this.message = message;
        }
    }

    public Message createMessage(String toEndpointId, String message){
        return new Message(toEndpointId, message);
    }

    public Message createMessage(String toEndpointId, Payload playload){
        return new Message(toEndpointId, new String(playload.asBytes(), StandardCharsets.UTF_8));
    }

    private BLEChat(){
        messages = new HashMap<String, ArrayList<Message>>();
    }

    private final static BLEChat instance = new BLEChat();
    private final static String SERVICE_ID = "comp5311-blechat";
    private final static String TAG = "blechat";
    private String name;

    private BLEChatMessageRoomHandler mMessageRoomClient = null;
    private BLEChatConnectionSearchHandler mConnecitonSearchClient = null;
    private HashMap<String, ArrayList<Message>> messages;

    @Override
    public void onEndpointDiscovered(ConnectionsActivity.Endpoint endpoint) {
        mConnecitonSearchClient.onEndpointDiscovered(endpoint);
    }

    @Override
    public void onConnectionInitiated(Endpoint endpoint, ConnectionInfo connectionInfo){
        mConnecitonSearchClient.onConnectionInitiated(endpoint, connectionInfo);
    }

    @Override
    public void onReceive(Endpoint endpoint, Payload payload){
        mMessageRoomClient.onReceive(endpoint, payload);
    }

    @Override
    public void onEndpointDisconnected(Endpoint endpoint){
        if(mMessageRoomClient != null)
            mMessageRoomClient.onEndpointDisconnected(endpoint);
    }

    @Override
    public void onEndpointLostConnection(String endpointId){
        if(mMessageRoomClient != null)
            mMessageRoomClient.onEndpointLostConnection(endpointId);
        if(mConnectionsClient != null)
            mConnecitonSearchClient.onEndpointLostConnection(endpointId);
    }

    public void send(Message message){

        messages.get(message.toEndpointId).add(message);

        mConnectionsClient
                .sendPayload(message.toEndpointId, Payload.fromBytes(message.message.getBytes()))
                .addOnFailureListener(
                        new OnFailureListener() {
                            @Override
                            public void onFailure(@NonNull Exception e) {
                                logW("sendPayload() failed.", e);
                            }
                        });
    }

    @CallSuper
    public void startAdvertising(){
        super.startAdvertising();
    }

    @CallSuper
    public void acceptConnection(final Endpoint endpoint) {
        super.acceptConnection(endpoint);
    }

    @CallSuper
    public void rejectConnection(Endpoint endpoint) {
        super.rejectConnection(endpoint);
    }

    @CallSuper
    public void startDiscovering() {
        super.startDiscovering();
    }

    @CallSuper
    public void connectToEndpoint(final Endpoint endpoint) {
        super.connectToEndpoint(endpoint);
    }

    @CallSuper
    public Set<Endpoint> getDiscoveredEndpoints(){
        return super.getDiscoveredEndpoints();
    }

    public void setConnecitonSearchClient(BLEChatConnectionSearchHandler mConnecitonSearchClient) {
        this.mConnecitonSearchClient = mConnecitonSearchClient;
    }

    public void setMessageRoomClient(BLEChatMessageRoomHandler mMessageRoomClient){
        this.mMessageRoomClient = mMessageRoomClient;
    }

    public ArrayList<Message> getMessages(String endpointId){
        ArrayList<Message> mes = messages.get(endpointId);
        if(mes == null){
            mes = new ArrayList<Message>();
            messages.put(endpointId, mes);
        }
        return mes;
    }

    public static BLEChat getInstance() {
        return instance;
    }

    public void setName(String name) {
        this.name = name;
    }

    @Override
    protected String getName() {
        return name;
    }

    @Override
    protected String getTAG() {
        return TAG;
    }

    @Override
    protected String getServiceId() {
        return SERVICE_ID;
    }

    @Override
    protected Strategy getStrategy() {
        return Strategy.P2P_CLUSTER;
    }

    public void stopAll(){
        stopAllEndpoints();
        stopAdvertising();
        stopDiscovering();
    }
}
