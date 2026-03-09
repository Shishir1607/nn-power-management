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
    reg [10:0] tv_f0[0:19],tv_f1[0:19],tv_f2[0:19],
               tv_f3[0:19],tv_f4[0:19];
    reg [1:0]  tv_label[0:19];
    integer    pass, fail, total, fid;

    initial begin
        fid = $fopen("testvectors.txt","r");
        for (i=0; i<20; i=i+1)
            $fscanf(fid,"%d %d %d %d %d %d\n",
                tv_f0[i],tv_f1[i],tv_f2[i],
                tv_f3[i],tv_f4[i],tv_label[i]);
        $fclose(fid);

        clk=0; rst=1; start=0;
        f0=0; f1=0; f2=0; f3=0; f4=0;
        pass=0; fail=0; total=0;

        repeat(4) @(posedge clk); #1;
        rst=0;
        repeat(2) @(posedge clk); #1;

        $display("=====================================================");
        $display("   Power MLP FSM - Behavioral Simulation");
        $display("=====================================================");
        $display(" # |  f0   f1   f2   f3   f4 | Exp | Got | Result");
        $display("---+---------------------------+-----+-----+-------");

        for (i=0; i<20; i=i+1) begin
            f0=tv_f0[i]; f1=tv_f1[i]; f2=tv_f2[i];
            f3=tv_f3[i]; f4=tv_f4[i];

            @(posedge clk); #1; start=1;
            @(posedge clk); #1; start=0;

            @(posedge valid);
            @(posedge clk); #1;
            @(posedge clk); #1;

            total = total + 1;
            if (mode == tv_label[i]) begin
                pass = pass + 1;
                $display("%2d | %4d %4d %4d %4d %4d | %1d | %1d | PASS",
                    i+1,f0,f1,f2,f3,f4,tv_label[i],mode);
            end else begin
                fail = fail + 1;
                $display("%2d | %4d %4d %4d %4d %4d | %1d | %1d | FAIL <---",
                    i+1,f0,f1,f2,f3,f4,tv_label[i],mode);
            end

            repeat(3) @(posedge clk);
        end

        $display("=====================================================");
        $display("  Total    : %0d", total);
        $display("  Pass     : %0d", pass);
        $display("  Fail     : %0d", fail);
        $display("  Accuracy : %0d%%", pass*100/total);
        $display("=====================================================");
        $finish;
    end
endmodule