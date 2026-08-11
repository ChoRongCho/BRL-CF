import { spawnSync } from 'node:child_process'

const minimum = [18, 12, 0]
const current = process.versions.node.split('.').map(Number)

function isSupportedNode(version) {
  for (let index = 0; index < minimum.length; index += 1) {
    if (version[index] > minimum[index]) {
      return true
    }

    if (version[index] < minimum[index]) {
      return false
    }
  }

  return true
}

if (!isSupportedNode(current)) {
  console.warn(
    `Skipping "quasar prepare": Node ${process.versions.node} is installed, but @quasar/app-webpack requires Node 18.12.0 or newer.`,
  )
  console.warn('Run the frontend with Node 20 LTS or another supported Node 18.12+ version.')
  process.exit(0)
}

const result = spawnSync('quasar', ['prepare'], {
  shell: true,
  stdio: 'inherit',
})

process.exit(result.status === null ? 1 : result.status)
