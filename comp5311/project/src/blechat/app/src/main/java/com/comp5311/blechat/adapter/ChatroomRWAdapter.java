package com.comp5311.blechat.adapter;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.comp5311.blechat.BLEChat;
import com.comp5311.blechat.R;

import java.util.ArrayList;

public class ChatroomRWAdapter extends RecyclerView.Adapter<ChatroomRWAdapter.ChatroomViewHolder> {

    private ArrayList<BLEChat.Message> messages;
    private final static int TYPE_OWNER = 0;
    private final static int TYPE_RECEIVED = 1;

    public class ChatroomViewHolder extends RecyclerView.ViewHolder {
        private ChatroomRWAdapter adapter;
        public ChatroomViewHolder(@NonNull View itemView){
            super(itemView);
        }
        public ChatroomRWAdapter.ChatroomViewHolder linkAdapter(ChatroomRWAdapter adapter){
            this.adapter = adapter;
            return this;
        };
    }

    private class OwnerViewHolder extends ChatroomViewHolder {
        TextView etChatroomOwnerMes;
        public OwnerViewHolder(@NonNull View itemView){
            super(itemView);
            etChatroomOwnerMes = itemView.findViewById(R.id.etChatroomOwnerMes);
        }
    }

    private class ReceivedViewHolder extends ChatroomViewHolder {
        TextView etChatroomReceivedMes;
        public ReceivedViewHolder(@NonNull View itemView){
            super(itemView);
            etChatroomReceivedMes = itemView.findViewById(R.id.etChatroomReceivedMes);
        }
    }

    public ChatroomRWAdapter(ArrayList<BLEChat.Message> messages){

        this.messages = new ArrayList<>(messages);
    }

    public void appendMessage(BLEChat.Message message){
        messages.add(message);
        notifyItemInserted(messages.size()-1);
    }

    @NonNull
    @Override
    public ChatroomViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {

        if(viewType == TYPE_OWNER){
            View view = LayoutInflater.from(parent.getContext())
                    .inflate(R.layout.activity_message_chat_owner_item, parent, false);
            return new OwnerViewHolder(view).linkAdapter(this);
        }

        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.activity_message_chat_received_item, parent, false);
        return new ReceivedViewHolder(view).linkAdapter(this);
    }

    @Override
    public void onBindViewHolder(@NonNull ChatroomViewHolder holder, int position) {
        BLEChat.Message message = messages.get(position);
        TextView tvMes = (message.toEndpointId != null)? ((OwnerViewHolder)holder).etChatroomOwnerMes: ((ReceivedViewHolder)holder).etChatroomReceivedMes;
        tvMes.setText(message.message);
    }

    @Override
    public int getItemCount() {
        return messages.size();
    }

    @Override
    public int getItemViewType(int position){
        if(messages.get(position).toEndpointId != null)
            return TYPE_OWNER;
        return TYPE_RECEIVED;
    }




}
