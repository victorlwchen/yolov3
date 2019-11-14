#!/bin/bash
if [ ! -z "$PASSWORD" ]; then
    echo 'root:'$PASSWORD | chpasswd
fi

exec /usr/sbin/sshd -D
