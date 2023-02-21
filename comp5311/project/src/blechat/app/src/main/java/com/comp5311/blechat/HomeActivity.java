package com.comp5311.blechat;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Color;
import android.os.Bundle;
import android.text.Editable;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

public class HomeActivity extends AppCompatActivity {

    Button btnStart;
    EditText etUsername;
    TextView tvDesc;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_home);
        etUsername = (EditText) findViewById(R.id.etUsername);
        tvDesc = (TextView) findViewById(R.id.tvDesc) ;

        btnStart = (Button) findViewById(R.id.btnStart);
        btnStart.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View v) {
                Editable editableUsername  = etUsername.getText();

                if(editableUsername.length() > -1){
                    Intent intent = new Intent(HomeActivity.this, SearchActivity.class);
                    intent.putExtra("username", editableUsername.toString());
                    startActivity(intent);
                }else{
                    tvDesc.setTextColor(Color.parseColor("#E1306C"));
                }
//                Intent intent = new Intent( HomeActivity.this, GameActivity.class);
//                startActivity(intent);
            }
        });


    }


}