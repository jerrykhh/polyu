package com.comp5311.blechat.adapter;

import android.content.Intent;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.comp5311.blechat.R;
import com.comp5311.blechat.SearchActivity;
import com.comp5311.blechat.nearby.ConnectionsActivity;

import java.util.ArrayList;

public class SearchConnectionResultRWAdapter extends RecyclerView.Adapter<SearchConnectionResultRWAdapter.ViewHolder>{

    public interface OnConnectClickHandler {
        void onEndpointConnection(ConnectionsActivity.Endpoint endpoint);
    }

    private ArrayList<ConnectionsActivity.Endpoint> searchConnectionResults;
    private OnConnectClickHandler mClickHandler;
    private String username;
    private final static int TYPE_HEADER = 0;
    private final static int TYPE_ITEM = 1;


    public SearchConnectionResultRWAdapter(ArrayList<ConnectionsActivity.Endpoint> searchConnectionResults, String username, OnConnectClickHandler clickHandler) {
        this.searchConnectionResults = searchConnectionResults;
        this.username = username;
        mClickHandler = clickHandler;
    }

    @NonNull
    @Override
    public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {

//        if(viewType == TYPE_HEADER){
//            View view = LayoutInflater.from(parent.getContext())
//                    .inflate(R.layout.activity_search_connection_header_item, parent, false);
//            return new HeaderViewHolder(view).linkAdapter(this);
//        }

        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.activity_search_connection_item, parent, false);
        return new ItemViewHolder(view).linkAdapter(this);
    }

    @Override
    public int getItemViewType(int position){
//        if(position == 0)
//            return TYPE_HEADER;
        return TYPE_ITEM;
    }

    @Override
    public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
//        Log.d("ble-enpoint-rc-position", position+"");
//        if(position == 0){
//            ((HeaderViewHolder)holder).tvCnntHeaderTitle.setText("Welcome, "+ username);
//        }else{
            ConnectionsActivity.Endpoint endpoint = searchConnectionResults.get(position);
            Log.d("ble-enpoint-rc",endpoint.getId());
            ((ItemViewHolder)holder).id = endpoint.getId();
            ((ItemViewHolder)holder).tvCnntUsername.setText(endpoint.getName());
//        }

    }

    @Override
    public int getItemCount() {
        return searchConnectionResults.size();
    }

    public class ViewHolder extends RecyclerView.ViewHolder {
        private SearchConnectionResultRWAdapter adapter;

        public ViewHolder(@NonNull View itemView) {
            super(itemView);
        }

        public ViewHolder linkAdapter(SearchConnectionResultRWAdapter adapter){
            this.adapter = adapter;
            return this;
        };
    }

    public class HeaderViewHolder extends ViewHolder {

        TextView tvCnntHeaderTitle;
        private SearchConnectionResultRWAdapter adapter;

        public HeaderViewHolder(@NonNull View itemView) {
            super(itemView);
            tvCnntHeaderTitle = itemView.findViewById(R.id.tvCnntHeaderTitle);
        }

    }

    public class ItemViewHolder extends ViewHolder {
        // on below line we are creating variable.

        TextView tvCnntUsername;
        Button btnCnnt;
        String id;
        private SearchConnectionResultRWAdapter adapter;

        public ItemViewHolder(@NonNull View itemView) {
            super(itemView);
            // on below line we are initialing our variable.

            tvCnntUsername = itemView.findViewById(R.id.tvCnntUsername);
            btnCnnt = itemView.findViewById(R.id.btnCnnt);
            btnCnnt.setOnClickListener(view -> {
//                Intent intent = new Intent(view.getContext(), SearchActivity.class);
//                intent.putExtra("endpointId", id);
//                view.getContext().startActivity(intent);
                mClickHandler.onEndpointConnection(searchConnectionResults.get(getAdapterPosition()));
            });
        }

        public void setId(String id){
            this.id = id;
        }
    }
}
