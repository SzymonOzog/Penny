n_pes = 24
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

        M = n_pes // gpus_per_node
        r1 = ((ring_id // 2) * 2 + 1) % gpus_per_node
        off = (gpus_per_node - r1) % gpus_per_node
        ring_pos = ((( -my_node) % M) * gpus_per_node + ((local_rank - r1) % gpus_per_node) - off) % n_pes

        color='\033[0m' #]
        if local_rank == (ring_id // 2) * 2:
            color = '\033[35m'#]

            send_peer = (n_pes + pe - gpus_per_node+1) % n_pes
            recv_peer = my_node * gpus_per_node + (local_rank - 1) % gpus_per_node

        elif local_rank == (ring_id // 2) * 2 + 1:
            color = '\033[36m'#]

            send_peer = my_node * gpus_per_node + (local_rank + 1) % gpus_per_node
            recv_peer = (n_pes + pe + gpus_per_node - 1) % n_pes

        else:
            send_peer = my_node * gpus_per_node + (local_rank + 1) % gpus_per_node
            recv_peer = my_node * gpus_per_node + (gpus_per_node + local_rank - 1) % gpus_per_node
        warning = ""

        send_chunk = (ring_pos)%n_pes
        recv_chunk = (ring_pos-1)%n_pes

        if len(csends) and csends[-1] != recv_chunk:
            warning += "*"

        if ring_pos != real_pos:
            warning += "#"

        sends.append(send_peer)
        recvs.append(recv_peer)
        csends.append(send_chunk)
        crecvs.append(recv_chunk)
        print(color, ring_pos, pe, " ", recv_peer, send_peer, " ", recv_chunk, send_chunk, warning)
        pe = send_peer
        real_pos +=1
    print('\033[0m')

