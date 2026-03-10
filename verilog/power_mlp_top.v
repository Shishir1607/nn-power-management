`timescale 1ns/1ps

// ============================================================
// Layer 1: 5 inputs → 8 neurons (ReLU)
// Weights scaled ×1000, inputs 11-bit unsigned
// All intermediate values: signed 48-bit
// ============================================================
module layer1 (
    input  wire        clk,
    input  wire        rst,
    input  wire        start,
    input  wire [10:0] f0, f1, f2, f3, f4,
    output reg  signed [47:0] h1_0, h1_1, h1_2, h1_3,
                               h1_4, h1_5, h1_6, h1_7,
    output reg         done
);
    // W1[neuron][input]
    reg signed [15:0] W1 [0:7][0:4];
    reg signed [47:0] B1 [0:7];

    integer ni;
    reg signed [47:0] acc;
    reg [3:0] state;
    reg [3:0] ii;

    localparam IDLE=0, MAC=1, BIAS=2, RELU=3, FINISH=4;

    initial begin
        W1[0][0]=-204;  W1[0][1]=834;   W1[0][2]=-676;  W1[0][3]=-1250; W1[0][4]=1566;
        W1[1][0]=729;   W1[1][1]=-275;  W1[1][2]=975;   W1[1][3]=2069;  W1[1][4]=335;
        W1[2][0]=-106;  W1[2][1]=465;   W1[2][2]=-212;  W1[2][3]=-1499; W1[2][4]=1485;
        W1[3][0]=285;   W1[3][1]=47;    W1[3][2]=497;   W1[3][3]=1124;  W1[3][4]=324;
        W1[4][0]=315;   W1[4][1]=-218;  W1[4][2]=406;   W1[4][3]=1833;  W1[4][4]=-8;
        W1[5][0]=-206;  W1[5][1]=-126;  W1[5][2]=-269;  W1[5][3]=42;    W1[5][4]=-442;
        W1[6][0]=-80;   W1[6][1]=4;     W1[6][2]=-186;  W1[6][3]=-1497; W1[6][4]=1179;
        W1[7][0]=393;   W1[7][1]=-199;  W1[7][2]=572;   W1[7][3]=1308;  W1[7][4]=464;

        B1[0]=1549626;  B1[1]=-76712;   B1[2]=1446509;  B1[3]=-165209;
        B1[4]=152380;   B1[5]=-195509;  B1[6]=1537917;  B1[7]=-89662;
    end

    reg signed [47:0] x0,x1,x2,x3,x4;

    always @(posedge clk) begin
        if (rst) begin
            state<=IDLE; done<=0; ni<=0; ii<=0; acc<=0;
        end else begin
            done <= 0;
            case (state)
                IDLE: begin
                    if (start) begin
                        x0<={37'b0,f0}; x1<={37'b0,f1};
                        x2<={37'b0,f2}; x3<={37'b0,f3};
                        x4<={37'b0,f4};
                        ni<=0; ii<=0; acc<=0; state<=MAC;
                    end
                end
                MAC: begin
                    case(ii)
                        0: acc <= acc + W1[ni][0]*x0;
                        1: acc <= acc + W1[ni][1]*x1;
                        2: acc <= acc + W1[ni][2]*x2;
                        3: acc <= acc + W1[ni][3]*x3;
                        4: acc <= acc + W1[ni][4]*x4;
                    endcase
                    if (ii==4) state<=BIAS;
                    else ii<=ii+1;
                end
                BIAS: begin
                    acc <= acc + B1[ni];
                    state <= RELU;
                end
                RELU: begin
                    case(ni)
                        0: h1_0 <= acc[47] ? 48'sd0 : (acc>>>10);
                        1: h1_1 <= acc[47] ? 48'sd0 : (acc>>>10);
                        2: h1_2 <= acc[47] ? 48'sd0 : (acc>>>10);
                        3: h1_3 <= acc[47] ? 48'sd0 : (acc>>>10);
                        4: h1_4 <= acc[47] ? 48'sd0 : (acc>>>10);
                        5: h1_5 <= acc[47] ? 48'sd0 : (acc>>>10);
                        6: h1_6 <= acc[47] ? 48'sd0 : (acc>>>10);
                        7: h1_7 <= acc[47] ? 48'sd0 : (acc>>>10);
                    endcase
                    acc <= 0; ii <= 0;
                    if (ni==7) state<=FINISH;
                    else begin ni<=ni+1; state<=MAC; end
                end
                FINISH: begin
                    done<=1; state<=IDLE;
                end
            endcase
        end
    end
endmodule


// ============================================================
// Layer 2: 8 inputs → 4 neurons (ReLU)
// ============================================================
module layer2 (
    input  wire        clk,
    input  wire        rst,
    input  wire        start,
    input  wire signed [47:0] h1_0,h1_1,h1_2,h1_3,
                               h1_4,h1_5,h1_6,h1_7,
    output reg  signed [47:0] h2_0, h2_1, h2_2, h2_3,
    output reg         done
);
    reg signed [15:0] W2 [0:3][0:7];
    reg signed [47:0] B2 [0:3];

    integer ni;
    reg signed [47:0] acc;
    reg [3:0] state;
    reg [3:0] ii;

    localparam IDLE=0, MAC=1, BIAS=2, RELU=3, FINISH=4;

    initial begin
        W2[0][0]=180;   W2[0][1]=-215;  W2[0][2]=-350;  W2[0][3]=-137;
        W2[0][4]=-271;  W2[0][5]=290;   W2[0][6]=102;   W2[0][7]=146;
        W2[1][0]=-41;   W2[1][1]=-6;    W2[1][2]=162;   W2[1][3]=-291;
        W2[1][4]=22;    W2[1][5]=-241;  W2[1][6]=-19;   W2[1][7]=-122;
        W2[2][0]=1766;  W2[2][1]=-1378; W2[2][2]=2416;  W2[2][3]=-624;
        W2[2][4]=-1240; W2[2][5]=-211;  W2[2][6]=2898;  W2[2][7]=-974;
        W2[3][0]=284;   W2[3][1]=-307;  W2[3][2]=-407;  W2[3][3]=-277;
        W2[3][4]=-238;  W2[3][5]=143;   W2[3][6]=70;    W2[3][7]=264;

        B2[0]=-182584;  B2[1]=-330481;
        B2[2]=1189913;  B2[3]=-198441;
    end

    reg signed [47:0] in0,in1,in2,in3,in4,in5,in6,in7;

    always @(posedge clk) begin
        if (rst) begin
            state<=IDLE; done<=0; ni<=0; ii<=0; acc<=0;
        end else begin
            done <= 0;
            case (state)
                IDLE: begin
                    if (start) begin
                        in0<=h1_0; in1<=h1_1; in2<=h1_2; in3<=h1_3;
                        in4<=h1_4; in5<=h1_5; in6<=h1_6; in7<=h1_7;
                        ni<=0; ii<=0; acc<=0; state<=MAC;
                    end
                end
                MAC: begin
                    case(ii)
                        0: acc <= acc + W2[ni][0]*in0;
                        1: acc <= acc + W2[ni][1]*in1;
                        2: acc <= acc + W2[ni][2]*in2;
                        3: acc <= acc + W2[ni][3]*in3;
                        4: acc <= acc + W2[ni][4]*in4;
                        5: acc <= acc + W2[ni][5]*in5;
                        6: acc <= acc + W2[ni][6]*in6;
                        7: acc <= acc + W2[ni][7]*in7;
                    endcase
                    if (ii==7) state<=BIAS;
                    else ii<=ii+1;
                end
                BIAS: begin
                    acc <= acc + B2[ni];
                    state <= RELU;
                end
                RELU: begin
                    case(ni)
                        0: h2_0 <= acc[47] ? 48'sd0 : (acc>>>10);
                        1: h2_1 <= acc[47] ? 48'sd0 : (acc>>>10);
                        2: h2_2 <= acc[47] ? 48'sd0 : (acc>>>10);
                        3: h2_3 <= acc[47] ? 48'sd0 : (acc>>>10);
                    endcase
                    acc <= 0; ii <= 0;
                    if (ni==3) state<=FINISH;
                    else begin ni<=ni+1; state<=MAC; end
                end
                FINISH: begin
                    done<=1; state<=IDLE;
                end
            endcase
        end
    end
endmodule


// ============================================================
// Layer 3: 4 inputs → 4 outputs (no activation, raw logits)
// ============================================================
module layer3 (
    input  wire        clk,
    input  wire        rst,
    input  wire        start,
    input  wire signed [47:0] h2_0, h2_1, h2_2, h2_3,
    output reg  [1:0]  mode,
    output reg         valid
);
    reg signed [15:0] W3 [0:3][0:3];
    reg signed [47:0] B3 [0:3];

    integer ni;
    reg signed [47:0] acc;
    reg [3:0] state;
    reg [3:0] ii;
    reg signed [47:0] z0,z1,z2,z3;

    localparam IDLE=0, MAC=1, BIAS=2, STORE=3, ARGMAX=4, FINISH=5;

    initial begin
        W3[0][0]=303;   W3[0][1]=-37;   W3[0][2]=1282;  W3[0][3]=-343;
        W3[1][0]=-252;  W3[1][1]=58;    W3[1][2]=784;   W3[1][3]=-172;
        W3[2][0]=298;   W3[2][1]=278;   W3[2][2]=-513;  W3[2][3]=-268;
        W3[3][0]=458;   W3[3][1]=-248;  W3[3][2]=-6086; W3[3][3]=-487;

        B3[0]=-7578354; B3[1]=-1792502;
        B3[2]=3147357;  B3[3]=5905848;
    end

    reg signed [47:0] in0,in1,in2,in3;

    always @(posedge clk) begin
        if (rst) begin
            state<=IDLE; valid<=0; ni<=0; ii<=0; acc<=0;
            z0<=0; z1<=0; z2<=0; z3<=0;
        end else begin
            valid <= 0;
            case (state)
                IDLE: begin
                    if (start) begin
                        in0<=h2_0; in1<=h2_1;
                        in2<=h2_2; in3<=h2_3;
                        ni<=0; ii<=0; acc<=0; state<=MAC;
                    end
                end
                MAC: begin
                    case(ii)
                        0: acc <= acc + W3[ni][0]*in0;
                        1: acc <= acc + W3[ni][1]*in1;
                        2: acc <= acc + W3[ni][2]*in2;
                        3: acc <= acc + W3[ni][3]*in3;
                    endcase
                    if (ii==3) state<=BIAS;
                    else ii<=ii+1;
                end
                BIAS: begin
                    acc <= acc + B3[ni];
                    state <= STORE;
                end
                STORE: begin
                    case(ni)
                        0: z0 <= acc;
                        1: z1 <= acc;
                        2: z2 <= acc;
                        3: z3 <= acc;
                    endcase
                    acc <= 0; ii <= 0;
                    if (ni==3) state<=ARGMAX;
                    else begin ni<=ni+1; state<=MAC; end
                end
                ARGMAX: begin
                    if      (z0>=z1 && z0>=z2 && z0>=z3) mode<=2'd0;
                    else if (z1>=z0 && z1>=z2 && z1>=z3) mode<=2'd1;
                    else if (z2>=z0 && z2>=z1 && z2>=z3) mode<=2'd2;
                    else                                  mode<=2'd3;
                    state<=FINISH;
                end
                FINISH: begin
                    valid<=1; state<=IDLE;
                end
            endcase
        end
    end
endmodule


// ============================================================
// Top Level: chains layer1 → layer2 → layer3
// ============================================================
module power_mlp_top (
    input  wire        clk,
    input  wire        rst,
    input  wire        start,
    input  wire [10:0] f0, f1, f2, f3, f4,
    output wire [1:0]  mode,
    output wire        valid
);
    wire l1_done, l2_done;
    wire signed [47:0] h1_0,h1_1,h1_2,h1_3,h1_4,h1_5,h1_6,h1_7;
    wire signed [47:0] h2_0,h2_1,h2_2,h2_3;

    layer1 u_l1 (
        .clk(clk), .rst(rst), .start(start),
        .f0(f0),.f1(f1),.f2(f2),.f3(f3),.f4(f4),
        .h1_0(h1_0),.h1_1(h1_1),.h1_2(h1_2),.h1_3(h1_3),
        .h1_4(h1_4),.h1_5(h1_5),.h1_6(h1_6),.h1_7(h1_7),
        .done(l1_done)
    );

    layer2 u_l2 (
        .clk(clk), .rst(rst), .start(l1_done),
        .h1_0(h1_0),.h1_1(h1_1),.h1_2(h1_2),.h1_3(h1_3),
        .h1_4(h1_4),.h1_5(h1_5),.h1_6(h1_6),.h1_7(h1_7),
        .h2_0(h2_0),.h2_1(h2_1),.h2_2(h2_2),.h2_3(h2_3),
        .done(l2_done)
    );

    layer3 u_l3 (
        .clk(clk), .rst(rst), .start(l2_done),
        .h2_0(h2_0),.h2_1(h2_1),.h2_2(h2_2),.h2_3(h2_3),
        .mode(mode), .valid(valid)
    );

endmodule
