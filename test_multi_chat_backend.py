import os
import sys
import sqlite3
import shutil
from unittest.mock import MagicMock

# Mock loguru
sys.modules["loguru"] = MagicMock()

from backend.core.storage.sqlite_manager import SQLiteStorageManager

# Constants
TEST_DB = "data/database/test_multi_chat.db"

def setup():
    if os.path.exists(TEST_DB):
        try:
            os.remove(TEST_DB)
        except:
            pass
    # Ensure dir
    os.makedirs(os.path.dirname(TEST_DB), exist_ok=True)

def test_multi_chat_flow():
    print(f"Initializing Storage Manager with {TEST_DB}...")
    manager = SQLiteStorageManager(db_path=TEST_DB)
    
    # 1. Create Chats
    print("\n1. Testing Chat Creation...")
    chat1 = manager.create_chat(title="Chat One")
    print(f"Created Chat 1: {chat1['id']} - {chat1['title']}")
    
    chat2 = manager.create_chat(title="Chat Two")
    print(f"Created Chat 2: {chat2['id']} - {chat2['title']}")
    
    chats = manager.get_chats()
    assert len(chats) == 2
    print("✅ Chat creation verified.")

    # 2. Add Messages
    print("\n2. Testing Message Isolation...")
    manager.save_chat_message(chat1['id'], "user", "Hello Chat 1")
    manager.save_chat_message(chat1['id'], "assistant", "Response 1")
    
    manager.save_chat_message(chat2['id'], "user", "Hello Chat 2")
    
    hist1 = manager.get_chat_history(chat1['id'])
    hist2 = manager.get_chat_history(chat2['id'])
    
    print(f"Chat 1 History Length: {len(hist1)}")
    print(f"Chat 2 History Length: {len(hist2)}")
    
    assert len(hist1) == 2
    assert len(hist2) == 1
    assert hist1[0]['content'] == "Hello Chat 1"
    assert hist2[0]['content'] == "Hello Chat 2"
    print("✅ Message isolation verified.")

    # 3. Rename Chat
    print("\n3. Testing Rename...")
    manager.rename_chat(chat1['id'], "Updated Chat One")
    updated_chat1 = manager.get_chat(chat1['id'])
    print(f"Renamed Chat 1 Title: {updated_chat1['title']}")
    assert updated_chat1['title'] == "Updated Chat One"
    print("✅ Chat rename verified.")

    # 4. Delete Chat
    print("\n4. Testing Delete...")
    manager.delete_chat(chat2['id'])
    chats_after = manager.get_chats()
    assert len(chats_after) == 1
    assert chats_after[0]['id'] == chat1['id']
    
    # Verify history deletion
    hist2_after = manager.get_chat_history(chat2['id'])
    assert len(hist2_after) == 0
    print("✅ Chat deletion verified.")

    print("\n🎉 All Multi-Chat Backend Tests Passed!")

if __name__ == "__main__":
    setup()
    try:
        test_multi_chat_flow()
    except AssertionError as e:
        print(f"❌ Test Failed: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
