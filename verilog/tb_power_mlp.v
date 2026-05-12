`timescale 1ns/1ps

module tb_power_mlp;

    reg         clk, rst, start;
    reg [10:0]  f0, f1, f2, f3, f4;
    wire [1:0]  mode;
    wire        valid;

    // DUT
    power_mlp_top dut (
        .clk(clk), .rst(rst), .start(start),
        .f0(f0),.f1(f1),.f2(f2),.f3(f3),.f4(f4),
        .mode(mode), .valid(valid)
    );

    // Clock: 10ns period
    always #5 clk = ~clk;

    // Test vector storage
    integer N;
    parameter MAX_N = 20100;
    reg [10:0] tv_f0[0:MAX_N-1], tv_f1[0:MAX_N-1],
               tv_f2[0:MAX_N-1], tv_f3[0:MAX_N-1],
               tv_f4[0:MAX_N-1];
    reg [1:0]  tv_exp[0:MAX_N-1];

    // Counters
    integer total, pass;
    integer s_tot, s_pass;
    integer lp_tot, lp_pass;
    integer b_tot,  b_pass;
    integer p_tot,  p_pass;
    integer i, ret;
    integer fh;

    initial begin
        clk=0; rst=1; start=0;
        f0=0; f1=0; f2=0; f3=0; f4=0;
        total=0; pass=0;
        s_tot=0;  s_pass=0;
        lp_tot=0; lp_pass=0;
        b_tot=0;  b_pass=0;
        p_tot=0;  p_pass=0;

        // ── Load test vectors ──────────────────────────────
        fh = $fopen("testvectors_20000.txt", "r");
        if (fh == 0) begin
            $display("ERROR: Could not open testvectors_20000.txt");
            $finish;
        end

        N = 0;
        while (!$feof(fh) && N < MAX_N) begin
            ret = $fscanf(fh, "%d %d %d %d %d %d\n",
                tv_f0[N], tv_f1[N], tv_f2[N],
                tv_f3[N], tv_f4[N], tv_exp[N]);
            if (ret == 6) N = N + 1;
        end
        $fclose(fh);
        $display("Loaded %0d test vectors", N);

        // ── Reset ─────────────────────────────────────────
        repeat(4) @(posedge clk);
        rst = 0;
        repeat(2) @(posedge clk);

        // ── Run all vectors ───────────────────────────────
        for (i = 0; i < N; i = i + 1) begin
            f0 = tv_f0[i]; f1 = tv_f1[i]; f2 = tv_f2[i];
            f3 = tv_f3[i]; f4 = tv_f4[i];

            @(posedge clk); #1;
            start = 1;
            @(posedge clk); #1;
            start = 0;

            // Wait for valid
            wait(valid == 1);
            @(posedge clk); #1;

            // Score
            total = total + 1;
            case(tv_exp[i])
                0: begin
                    s_tot = s_tot + 1;
                    if (mode == tv_exp[i]) begin
                        pass = pass + 1;
                        s_pass = s_pass + 1;
                    end
                end
                1: begin
                    lp_tot = lp_tot + 1;
                    if (mode == tv_exp[i]) begin
                        pass = pass + 1;
                        lp_pass = lp_pass + 1;
                    end
                end
                2: begin
                    b_tot = b_tot + 1;
                    if (mode == tv_exp[i]) begin
                        pass = pass + 1;
                        b_pass = b_pass + 1;
                    end
                end
                3: begin
                    p_tot = p_tot + 1;
                    if (mode == tv_exp[i]) begin
                        pass = pass + 1;
                        p_pass = p_pass + 1;
                    end
                end
            endcase

            repeat(2) @(posedge clk);
        end

        // ── Final Report ──────────────────────────────────
        $display("=====================================================");
        $display("   Power MLP FSM - 20000 Augmented Samples");
        $display("=====================================================");
        $display("  Sleep       : %0d/%0d (%0d%%)",
            s_pass,  s_tot,  (s_tot  ? s_pass*100/s_tot   : 0));
        $display("  LowPower    : %0d/%0d (%0d%%)",
            lp_pass, lp_tot, (lp_tot ? lp_pass*100/lp_tot : 0));
        $display("  Balanced    : %0d/%0d (%0d%%)",
            b_pass,  b_tot,  (b_tot  ? b_pass*100/b_tot   : 0));
        $display("  Performance : %0d/%0d (%0d%%)",
            p_pass,  p_tot,  (p_tot  ? p_pass*100/p_tot   : 0));
        $display("-----------------------------------------------------");
        $display("  Overall     : %0d/%0d (%0d%%)",
            pass, total, (total ? pass*100/total : 0));
        $display("=====================================================");
        $finish;
    end

endmodule