# Raspberry Pi 設定

## Welcome to Raspberry Pi 的設定

1. **初始畫面**
   1. 按 `Next`。
2. **Set Country**
   1. Country 選 `Taiwan`，Language 和 Timezone 跟著一起變成 Chinese 和 Taipei 之後就不用動了。
   2. 把 Use English language 和 Use US keyboard 勾起來
   3. 按 `Next`。
3. **Change Password**
   1. 改完密碼後按 `Next`。（如果不改的話，預設密碼是`raspberry`。）
4. **Setup Screen**
   1. 如果螢幕沒問題就按 `Next`。（如果螢幕有出現黑邊，就把方框勾起來。）
5. **Set WiFi Network**
   1. 直接 `Skip`。（另外設定就好）
6. **Update Software**
   1. 直接 `Skip`。（等更新要等很久）
7. 按 `Done` 就好。

## 網路和 Anydesk 的設定

1. 連上 WiFi（右上角）
2. 打開瀏覽器（左上角的地球）下載 Anydesk 的 Raspberry Pi 版本的 DEB Package
3. 安裝後執行 Anydesk
   - （左上角的樹莓圖案） >> Internet >> Anydesk
4. 左邊有個 Set password for unattended access...
5. 進去之後選有個盾牌，後面寫 Unlock Security Settings，點他
6. 選取「啟用無人值守存取」（unattended access）
7. 設定完密碼後，紀錄 This Desk 下面的 IP

## 改 Username 和 Resolution

1. （左上角的樹莓圖案） >> Preference >> Raspberry PI Configuration
   1. 把 Auto login: 改成 Disable
   2. 把 Network at boot: 改成 Wait for network
2. Preference >> Raspberry PI Configuration >> Display >> Set Resolution 改成 `DMT mode 82 1920 x 1080`。
   - 改完會問你要不要 reboot，選 No
3. 開啟 Terminal （左上角）用 `sudo passwd root` 指定新密碼給 root
4. `reboot` 重開機後選 other user，用 root 登入
5. 用 root 登入的話上面的工具列會不見。用 Ctrl+Alt+T 打開 terminal
6. `usermod -l newname oldname`（newname放是要改新名字，oldname 放舊名字，預設應該都是 `pi`）
7. `usermod -m -d /home/newname newname`
8. `reboot` 重開機後選 newname 登入
9. Preference >> RaspberryPI Configuration >> 把 Auto login: 改回 Login as user 'pi'

- 更新 code

```sh
git clone https://github.com/Andrew-mie7/bee_sound.git
cd bee_sound
sudo sh setup_sudo.sh
sh setup_no_sudo.sh
reboot
```

```sh
# get the info of Mic Capture Volume
amixer --card 2 cget numid=8
# set the value of Mic Capture Volume
amixer --card 2 cset numid=8 [value]
```