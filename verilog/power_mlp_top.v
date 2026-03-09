`timescale 1ns/1ps
module power_mlp_top (
    input  wire        clk,
    input  wire        rst,
    input  wire        start,
    input  wire [10:0] f0, f1, f2, f3, f4,
    output reg  [1:0]  mode,
    output reg         valid
);
    localparam IDLE   = 4'd0,
               L1_MAC = 4'd1,
               L1_BIAS= 4'd2,
               L1_RELU= 4'd3,
               L2_MAC = 4'd4,
               L2_BIAS= 4'd5,
               L2_RELU= 4'd6,
               L3_MAC = 4'd7,
               L3_BIAS= 4'd8,
               OUTPUT = 4'd9;

    reg [3:0]  state;
    reg [3:0]  ni;
    reg [3:0]  ii;

    reg signed [47:0] acc;
    reg signed [47:0] h1 [0:7];
    reg signed [47:0] h2 [0:3];
    reg signed [47:0] z3 [0:3];
    reg signed [15:0] x  [0:4];

    reg signed [15:0] W1 [0:7][0:4];
    reg signed [15:0] W2 [0:3][0:7];
    reg signed [15:0] W3 [0:3][0:3];
    reg signed [47:0] B1 [0:7];
    reg signed [47:0] B2 [0:3];
    reg signed [47:0] B3 [0:3];

    integer k;

    initial begin
        W1[0][0]=-204;  W1[0][1]=834;   W1[0][2]=-676;
        W1[0][3]=-1250; W1[0][4]=1566;
        W1[1][0]=729;   W1[1][1]=-275;  W1[1][2]=975;
        W1[1][3]=2069;  W1[1][4]=335;
        W1[2][0]=-106;  W1[2][1]=465;   W1[2][2]=-212;
        W1[2][3]=-1499; W1[2][4]=1485;
        W1[3][0]=285;   W1[3][1]=47;    W1[3][2]=497;
        W1[3][3]=1124;  W1[3][4]=324;
        W1[4][0]=315;   W1[4][1]=-218;  W1[4][2]=406;
        W1[4][3]=1833;  W1[4][4]=-8;
        W1[5][0]=-206;  W1[5][1]=-126;  W1[5][2]=-269;
        W1[5][3]=42;    W1[5][4]=-442;
        W1[6][0]=-80;   W1[6][1]=4;     W1[6][2]=-186;
        W1[6][3]=-1497; W1[6][4]=1179;
        W1[7][0]=393;   W1[7][1]=-199;  W1[7][2]=572;
        W1[7][3]=1308;  W1[7][4]=464;

        B1[0]=1549626;  B1[1]=-76712;   B1[2]=1446509;
        B1[3]=-165209;  B1[4]=152380;   B1[5]=-195509;
        B1[6]=1537917;  B1[7]=-89662;

        W2[0][0]=180;   W2[0][1]=-215;  W2[0][2]=-350;
        W2[0][3]=-137;  W2[0][4]=-271;  W2[0][5]=290;
        W2[0][6]=102;   W2[0][7]=146;
        W2[1][0]=-41;   W2[1][1]=-6;    W2[1][2]=162;
        W2[1][3]=-291;  W2[1][4]=22;    W2[1][5]=-241;
        W2[1][6]=-19;   W2[1][7]=-122;
        W2[2][0]=1766;  W2[2][1]=-1378; W2[2][2]=2416;
        W2[2][3]=-624;  W2[2][4]=-1240; W2[2][5]=-211;
        W2[2][6]=2898;  W2[2][7]=-974;
        W2[3][0]=284;   W2[3][1]=-307;  W2[3][2]=-407;
        W2[3][3]=-277;  W2[3][4]=-238;  W2[3][5]=143;
        W2[3][6]=70;    W2[3][7]=264;

        B2[0]=-182584;  B2[1]=-330481;
        B2[2]=1189913;  B2[3]=-198441;

        W3[0][0]=303;   W3[0][1]=-37;   W3[0][2]=1282;  W3[0][3]=-343;
        W3[1][0]=-252;  W3[1][1]=58;    W3[1][2]=784;   W3[1][3]=-172;
        W3[2][0]=298;   W3[2][1]=278;   W3[2][2]=-513;  W3[2][3]=-268;
        W3[3][0]=458;   W3[3][1]=-248;  W3[3][2]=-6086; W3[3][3]=-487;

        B3[0]=-7578354; B3[1]=-1792502;
        B3[2]=3147357;  B3[3]=5905848;
    end

    always @(posedge clk) begin
        if (rst) begin
            state <= IDLE; valid <= 0; mode <= 0;
            ni <= 0; ii <= 0; acc <= 48'sd0;
            for (k=0; k<8; k=k+1) h1[k] <= 0;
            for (k=0; k<4; k=k+1) h2[k] <= 0;
            for (k=0; k<4; k=k+1) z3[k] <= 0;
        end else begin
            case (state)

                IDLE: begin
                    valid <= 0;
                    if (start) begin
                        x[0] <= {5'b0, f0};
                        x[1] <= {5'b0, f1};
                        x[2] <= {5'b0, f2};
                        x[3] <= {5'b0, f3};
                        x[4] <= {5'b0, f4};
                        ni <= 0; ii <= 0;
                        acc <= 48'sd0;
                        state <= L1_MAC;
                    end
                end

                L1_MAC: begin
                    if (ii == 5) begin
                        ii <= 0; state <= L1_BIAS;
                    end else begin
                        acc <= acc + ($signed(W1[ni][ii]) * $signed(x[ii]));
                        ii <= ii + 1;
                    end
                end

                L1_BIAS: begin
                    acc   <= acc + B1[ni];
                    state <= L1_RELU;
                end

                L1_RELU: begin
                    h1[ni] <= (acc[47]) ? 48'sd0 : (acc >>> 10);
                    acc    <= 48'sd0;
                    if (ni == 7) begin
                        ni <= 0; state <= L2_MAC;
                    end else begin
                        ni <= ni + 1; state <= L1_MAC;
                    end
                end

                L2_MAC: begin
                    if (ii == 8) begin
                        ii <= 0; state <= L2_BIAS;
                    end else begin
                        acc <= acc + ($signed(W2[ni][ii]) * $signed(h1[ii]));
                        ii <= ii + 1;
                    end
                end

                L2_BIAS: begin
                    acc   <= acc + B2[ni];
                    state <= L2_RELU;
                end

                L2_RELU: begin
                    h2[ni] <= (acc[47]) ? 48'sd0 : (acc >>> 10);
                    acc    <= 48'sd0;
                    if (ni == 3) begin
                        ni <= 0; state <= L3_MAC;
                    end else begin
                        ni <= ni + 1; state <= L2_MAC;
                    end
                end

                L3_MAC: begin
                    if (ii == 4) begin
                        ii <= 0; state <= L3_BIAS;
                    end else begin
                        acc <= acc + ($signed(W3[ni][ii]) * $signed(h2[ii]));
                        ii <= ii + 1;
                    end
                end

                L3_BIAS: begin
                    z3[ni] <= acc + B3[ni];
                    acc    <= 48'sd0;
                    if (ni == 3) begin
                        state <= OUTPUT;
                    end else begin
                        ni <= ni + 1; state <= L3_MAC;
                    end
                end

                OUTPUT: begin
                    if      (z3[0]>=z3[1] && z3[0]>=z3[2] && z3[0]>=z3[3]) mode<=2'd0;
                    else if (z3[1]>=z3[0] && z3[1]>=z3[2] && z3[1]>=z3[3]) mode<=2'd1;
                    else if (z3[2]>=z3[0] && z3[2]>=z3[1] && z3[2]>=z3[3]) mode<=2'd2;
                    else                                                      mode<=2'd3;
                    valid <= 1;
                    state <= IDLE;
                end

            endcase
        end
    end
endmodule