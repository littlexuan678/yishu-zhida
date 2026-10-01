<template>
  <header class="app-header">
    <div class="app-header__left">
      <el-icon class="app-header__badge" :size="20"><FirstAidKit /></el-icon>
      <h1 class="app-header__title">智愈医典·医疗知识信息问答系统</h1>
    </div>

    <div class="app-header__right">
      <span v-if="store.offline" class="app-header__offline">
        <el-icon><WarningFilled /></el-icon>
        离线演示数据
      </span>
      <el-button class="app-header__logout" type="danger" :icon="SwitchButton" @click="handleLogout">
        安全退出
      </el-button>
    </div>
  </header>
</template>

<script setup>
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { SwitchButton, FirstAidKit, WarningFilled } from '@element-plus/icons-vue'
import { useAppStore } from '@/store/app'

const router = useRouter()
const store = useAppStore()

async function handleLogout () {
  try {
    await ElMessageBox.confirm('确定要退出当前登录吗？', '安全退出', {
      confirmButtonText: '确定退出',
      cancelButtonText: '取消',
      type: 'warning'
    })
  } catch (e) {
    return
  }
  store.resetSession()
  ElMessage.success('已安全退出，会话已重置')
  router.push('/')
}
</script>

<style scoped lang="scss">
.app-header {
  height: var(--mkw-header-height);
  background: #ffffff;
  border-bottom: 1px solid #e8eefb;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
  box-shadow: 0 1px 6px rgba(80, 120, 220, 0.05);
  position: sticky;
  top: 0;
  z-index: 20;
}

.app-header__left {
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 0;
}

.app-header__badge {
  color: var(--mkw-primary);
  flex: 0 0 auto;
}

.app-header__title {
  margin: 0;
  font-size: 19px;
  font-weight: 800;
  color: #2b8fe8;
  letter-spacing: 0.6px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.app-header__right {
  display: flex;
  align-items: center;
  gap: 12px;
  flex: 0 0 auto;
}

.app-header__offline {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-size: 12px;
  color: #c98b1e;
  background: #fdf5e6;
  border-radius: 20px;
  padding: 3px 10px;
}

.app-header__logout {
  border-radius: 10px;
  font-weight: 600;
  padding: 10px 18px;
  background: #f56c6c;
  border-color: #f56c6c;
  box-shadow: 0 4px 12px rgba(245, 108, 108, 0.28);
}

.app-header__logout:hover {
  background: #f78989;
  border-color: #f78989;
}

@media (max-width: 900px) {
  .app-header {
    padding: 0 12px;
  }

  .app-header__title {
    font-size: 15px;
  }
}
</style>
