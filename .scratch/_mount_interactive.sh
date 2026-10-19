C=~/.local/bin/colab
rm -f /tmp/drive_go ~/drive_url.txt
$C new -s mtest --gpu T4 || exit 1
( while [ ! -f /tmp/drive_go ]; do sleep 1; done; echo ) | PYTHONUNBUFFERED=1 $C drivemount -s mtest
echo "DRIVEMOUNT_RC=$?"
$C exec -s mtest -f /mnt/e/ENVEDA_casmi/research/queue/_mount_probe.py --timeout 300
$C stop -s mtest
$C sessions
