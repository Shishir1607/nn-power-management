`timescale 1ns/1ps
module tb_power_mlp;

    reg        clk, rst, start;
    reg [10:0] f0,f1,f2,f3,f4;
    wire [1:0] mode;
    wire       valid;

    power_mlp_top dut (
        .clk(clk), .rst(rst), .start(start),
        .f0(f0),.f1(f1),.f2(f2),.f3(f3),.f4(f4),
        .mode(mode), .valid(valid)
    );

    always #5 clk = ~clk;

    integer i;
    reg [10:0] tv_f0[0:6325],tv_f1[0:6325],tv_f2[0:6325],
               tv_f3[0:6325],tv_f4[0:6325];
    reg [1:0]  tv_label[0:6325];
    integer    pass, fail, total, fid;
    integer    pass0,pass1,pass2,pass3;
    integer    tot0,tot1,tot2,tot3;

    initial begin
        fid = $fopen("testvectors_real.txt","r");
        for (i=0; i<6326; i=i+1)
            $fscanf(fid,"%d %d %d %d %d %d\n",
                tv_f0[i],tv_f1[i],tv_f2[i],
                tv_f3[i],tv_f4[i],tv_label[i]);
        $fclose(fid);

        clk=0; rst=1; start=0;
        f0=0; f1=0; f2=0; f3=0; f4=0;
        pass=0; fail=0; total=0;
        pass0=0; pass1=0; pass2=0; pass3=0;
        tot0=0;  tot1=0;  tot2=0;  tot3=0;

        repeat(4) @(posedge clk); #1;
        rst=0;
        repeat(2) @(posedge clk); #1;

        for (i=0; i<6326; i=i+1) begin
            f0=tv_f0[i]; f1=tv_f1[i]; f2=tv_f2[i];
            f3=tv_f3[i]; f4=tv_f4[i];

            @(posedge clk); #1; start=1;
            @(posedge clk); #1; start=0;

            @(posedge valid);
            @(posedge clk); #1;
            @(posedge clk); #1;

            total = total + 1;

            case(tv_label[i])
                0: tot0=tot0+1;
                1: tot1=tot1+1;
                2: tot2=tot2+1;
                3: tot3=tot3+1;
            endcase

            if (mode == tv_label[i]) begin
                pass = pass + 1;
                case(tv_label[i])
                    0: pass0=pass0+1;
                    1: pass1=pass1+1;
                    2: pass2=pass2+1;
                    3: pass3=pass3+1;
                endcase
            end else
                fail = fail + 1;

            repeat(3) @(posedge clk);
        end

        $display("=====================================================");
        $display("   Power MLP FSM - 6326 Real CSV Samples");
        $display("=====================================================");
        $display("  Sleep       : %0d/%0d (%0d%%)", pass0,tot0, tot0>0 ? pass0*100/tot0 : 0);
        $display("  LowPower    : %0d/%0d (%0d%%)", pass1,tot1, tot1>0 ? pass1*100/tot1 : 0);
        $display("  Balanced    : %0d/%0d (%0d%%)", pass2,tot2, tot2>0 ? pass2*100/tot2 : 0);
        $display("  Performance : %0d/%0d (%0d%%)", pass3,tot3, tot3>0 ? pass3*100/tot3 : 0);
        $display("-----------------------------------------------------");
        $display("  Overall     : %0d/%0d (%0d%%)", pass,total, pass*100/total);
        $display("=====================================================");
        $finish;
    end
endmodule
