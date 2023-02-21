package com.comp5311.blechat;

import androidx.appcompat.widget.Toolbar;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import android.content.Intent;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import com.comp5311.blechat.adapter.SearchConnectionResultRWAdapter;
import com.comp5311.blechat.nearby.ConnectionsActivity;
import com.google.android.gms.nearby.connection.ConnectionInfo;
import com.google.android.gms.nearby.connection.Strategy;
import com.google.android.material.appbar.CollapsingToolbarLayout;
import com.google.android.material.bottomsheet.BottomSheetDialog;

import java.util.ArrayList;
import java.util.HashSet;

public class SearchActivity extends BLEChatActivity implements SearchConnectionResultRWAdapter.OnConnectClickHandler, BLEChat.BLEChatConnectionSearchHandler {

    private Toolbar toolbar;
    private RecyclerView rcUsers;
    private CollapsingToolbarLayout collapsingToolbar;
    private String username;
    private BLEChat bleChat;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_search);
        toolbar = (Toolbar) findViewById(R.id.toolbar);

        setSupportActionBar(toolbar);
        getSupportActionBar().setDisplayHomeAsUpEnabled(true);
        getSupportActionBar().setDisplayShowHomeEnabled(true);


        rcUsers = (RecyclerView) findViewById(R.id.rcUsers);
        Intent intent = getIntent();
        username = intent.getStringExtra("username");
        collapsingToolbar = (CollapsingToolbarLayout)findViewById(R.id.collapsingToolbar);

        collapsingToolbar.setTitle("Welcome, " + username);

        bleChat = BLEChat.getInstance();
        bleChat.setName(username);
        bleChat.setConnecitonSearchClient(this);
        bleChat.setActivity(this);

        // start
        bleChat.startAdvertising();
        bleChat.startDiscovering();


    }

    private void updateRCUsers(){
        ArrayList<ConnectionsActivity.Endpoint> endpoints = new ArrayList<>(bleChat.getDiscoveredEndpoints());
        rcUsers.setLayoutManager(new LinearLayoutManager(this));
        SearchConnectionResultRWAdapter adapter = new SearchConnectionResultRWAdapter(endpoints, username, this);
        rcUsers.setAdapter(adapter);
    }

    @Override
    public void onEndpointDiscovered(ConnectionsActivity.Endpoint endpoint) {
        updateRCUsers();
    }


    @Override
    public void onConnectionInitiated(ConnectionsActivity.Endpoint endpoint, ConnectionInfo connectionInfo){
        final BottomSheetDialog bottomSheetDialog = new BottomSheetDialog(SearchActivity.this, R.style.BottomSheetDialogTheme);
        View bottomSheetView = LayoutInflater.from(getApplicationContext())
                .inflate(
                        R.layout.activity_search_connection_modal,
                        (LinearLayout)findViewById(R.id.llModalContainer)
                );
        ((TextView)bottomSheetView.findViewById(R.id.tvNewCnntDesc)).setText("Request to connect with " + endpoint.getName());
        bottomSheetView.findViewById(R.id.btnCnntAccept).setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View view){
//                Toast.makeText(SearchActivity.this,"Click accept", Toast.LENGTH_SHORT).show();
                bleChat.acceptConnection(endpoint);
                Intent intent = new Intent(SearchActivity.this, MessageChatActivity.class);
                intent.putExtra("endpointId", endpoint.getId());
                startActivity(intent);
                bottomSheetDialog.hide();
            }
        });
        bottomSheetView.findViewById(R.id.btnCnntDecline).setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View view){
                bleChat.rejectConnection(endpoint);
                bottomSheetDialog.hide();
            }
        });
        bottomSheetDialog.setContentView(bottomSheetView);
        bottomSheetDialog.show();

    }

    @Override
    public void onEndpointLostConnection(String endpoint) {
        updateRCUsers();
    }


    @Override
    public void onEndpointConnection(ConnectionsActivity.Endpoint endpoint) {
        bleChat.connectToEndpoint(endpoint);
    }

}

