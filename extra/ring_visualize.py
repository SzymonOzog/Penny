n_pes = 16
gpus_per_node = 8


for ring_id in range(8):
    print(ring_id)
    sends = []
    recvs = []
    csends = []
    crecvs = []
    pes = []
    pe = 0
    real_pos = 0
    while pe not in pes:
        pes.append(pe)
        local_rank = pe%gpus_per_node
        my_node = pe//gpus_per_node

        ring_pos = real_pos

        color='\033[0m' #]
        # SENDER
        if local_rank == (ring_id // 2) * 2:
            color = '\033[35m'#]
            if my_node%2 == 1:
                send_peer = my_node * gpus_per_node + (gpus_per_node + local_rank - 1) % gpus_per_node
                recv_peer = (n_pes + pe - gpus_per_node) % n_pes
            else:
                send_peer = (n_pes + pe + gpus_per_node) % n_pes
                recv_peer = my_node * gpus_per_node + (gpus_per_node + local_rank - 1) % gpus_per_node

        #RECEIVER
        elif local_rank == (ring_id // 2) * 2 + 1:
            color = '\033[36m'#]

            if my_node%2 == 1:
                send_peer = (n_pes + pe + gpus_per_node) % n_pes
                recv_peer = my_node * gpus_per_node + (local_rank + 1) % gpus_per_node
            else:
                send_peer = my_node * gpus_per_node + (local_rank + 1) % gpus_per_node
                recv_peer = (n_pes + pe - gpus_per_node) % n_pes

        else:
            send_peer = my_node * gpus_per_node + (local_rank + 1) % gpus_per_node
            recv_peer = my_node * gpus_per_node + (gpus_per_node + local_rank - 1) % gpus_per_node
            if my_node%2 == 1:
                send_peer, recv_peer = recv_peer, send_peer
        warning = ""

        send_chunk = (ring_pos)%n_pes
        recv_chunk = (ring_pos-1)%n_pes

        if ring_id%2 == 1:
            send_peer, recv_peer = recv_peer, send_peer
            send_chunk, recv_chunk = recv_chunk, send_chunk

        local_csends = []
        local_crecvs = []
        for chunk in range(n_pes):
            local_crecvs.append(recv_chunk)
            local_csends.append(send_chunk)
            send_chunk = recv_chunk
            if ring_id%2 == 1:
                recv_chunk = (n_pes + recv_chunk + 1)%n_pes;
            else:
                recv_chunk = (n_pes + recv_chunk - 1)%n_pes;
        for chunk in range(n_pes):
            local_crecvs.append(recv_chunk)
            local_csends.append(send_chunk)
            send_chunk = recv_chunk
            if ring_id%2 == 1:
                recv_chunk = (n_pes + recv_chunk + 1)%n_pes;
            else:
                recv_chunk = (n_pes + recv_chunk - 1)%n_pes;
            

        if len(csends) and not all(x == y for x, y in zip(local_crecvs, csends[-1])):
            warning += "*"

        if ring_pos != real_pos:
            warning += "#"

        sends.append(send_peer)
        recvs.append(recv_peer)
        csends.append(local_csends)
        crecvs.append(local_crecvs)
        print(color, pe, " ", recv_peer, send_peer, " ", local_csends[0], local_crecvs[0], warning)
        pe = send_peer
        if ring_id%2 == 1:
            real_pos =(n_pes + real_pos - 1)%n_pes
        else:
            real_pos +=1
    print('\033[0m')

